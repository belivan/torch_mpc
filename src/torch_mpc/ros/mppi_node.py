#!/usr/bin/python3

import rospy
import torch
import numpy as np
import time
import yaml
import tf2_ros

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float32
from geometry_msgs.msg import Pose, Point, Vector3, Vector3Stamped, Quaternion, PoseArray
from sensor_msgs.msg import Joy
from nav_msgs.msg import OccupancyGrid, Odometry
from grid_map_msgs.msg import GridMap
from torch_mpc.msg import KBMParameters, MPPIStats, SteerSetpointKBMState

from torch_mpc.util.utils import dict_map
from torch_mpc.ros_converter.ros_converter import ROSConverter, pp_torch_dict
from torch_mpc.models.actuator_delay import ActuatorDelay
from torch_mpc.setup_mpc import setup_mpc

"""
Wrapper class around MPPI that does the following:
    1. Listens to ROS to get the current vehicle state
    2. Listens to ROS to get current cost function params
    3. Optimizes a sequence of actions using MPPI
    4. (Asynchronously) publishes viz
"""

class MPPINode:
    def __init__(self, model, cost_fn, mppi, dt, converter, base_frame, mppi_itrs):
        """
        Args:
            model: The model to optimize through
            cost_fn: The cost function to optimize
            mppi: The MPPI instance
            dt: The interval to sample MPPI commands
            converter: the OnlineConverter object to query current state from
            base_frame: the frame in which MPPI operates
            mppi_itrs: the number of mppi iterations per step
        """
        self.model = model
        self.cost_fn = cost_fn
        self.mppi = mppi
        self.dt = dt
        self.converter = converter
        self.rate = rospy.Rate(1/self.dt)
        self.base_frame = base_frame
        self.u = torch.zeros(self.mppi.m)
        self.u_seq = torch.zeros(self.mppi.H, self.mppi.m)
        self.alpha = 0.8
        self.mppi_itrs = mppi_itrs

        self.use_actuator_delay = isinstance(self.model, ActuatorDelay)
        rospy.loginfo('Modeling actuator delay = {}'.format(self.use_actuator_delay))

        #Update MPPI
        rospy.Timer(rospy.Duration(0.2), self.viz)
        self.viz_pub = rospy.Publisher("/mppi/viz", MarkerArray, queue_size=1)
        self.stats_pub = rospy.Publisher("/mppi/stats", MPPIStats, queue_size=1)
        self.cmd_pub = rospy.Publisher("/joy_auto", Joy, queue_size=1)
        self.vel_pub = rospy.Publisher("/controller/target_input", Float32, queue_size=1)

    def viz(self, event):
        """
        Plot the following:
            1. The current optimized state sequence
            2. A subset of the sampled trajs
            3. The current cost
        """
        marker_msg = MarkerArray()

        #Get optimized states
        best_idx = self.mppi.costs[0].argmin()
        last_states_msg = Marker()
        last_states_msg.header.stamp = rospy.Time.now()
        last_states_msg.header.frame_id = self.base_frame
        last_states_msg.ns = "mppi"
        last_states_msg.id = 0
        last_states_msg.type = Marker.LINE_STRIP
        last_states_msg.action = Marker.ADD
        last_states_msg.pose = Pose(orientation=Quaternion(w=1.0))
        last_states_msg.scale = Vector3(x=0.4, y=0.4, z=0.4)
#        last_states_msg.color = ColorRGBA(r=1., g=0., b=0., a=1.)
        last_states_msg.lifetime = rospy.Duration(0.5)
        last_states_msg.frame_locked=True
        last_states_msg.points, last_states_msg.colors = self.traj_to_pointarray(self.mppi.noisy_states[0, best_idx])
        marker_msg.markers.append(last_states_msg)

        #plot bbox
        bbox_msg = Marker()
        bbox_msg.header.stamp = rospy.Time.now()
        bbox_msg.header.frame_id = self.base_frame
        bbox_msg.ns = "mppi"
        bbox_msg.id = 1
        bbox_msg.type = Marker.LINE_STRIP
        bbox_msg.action = Marker.ADD
        bbox_msg.pose = Pose(
            position = Point(x=self.mppi.noisy_states[0, best_idx, 0, 0], y=self.mppi.noisy_states[0, best_idx, 0, 1]),
            orientation = self.yaw_to_quat(self.mppi.noisy_states[0, best_idx, 0, 2])
        )
        bbox_msg.scale = Vector3(x=0.2, y=0.2, z=0.2)
        bbox_msg.color = ColorRGBA(r=1., g=0., b=0., a=1.)
        bbox_msg.lifetime = rospy.Duration(0.5)
        bbox_msg.frame_locked=True
        bbox_msg.points, _ = self.traj_to_pointarray(torch.tensor([
            [-3.5, -1.5, 0, 0],
            [-3.5, 1.5, 0, 0],
            [1.5, 1.5, 0, 0],
            [1.5, -1.5, 0, 0],
            [-3.5, -1.5, 0, 0]
        ]))
        marker_msg.markers.append(bbox_msg)

        #get cost
        last_cost_msg = Marker()
        last_cost_msg.header.stamp = rospy.Time.now()
        last_cost_msg.header.frame_id = self.base_frame
        last_cost_msg.ns = "mppi"
        last_cost_msg.id = 2
        last_cost_msg.type = Marker.TEXT_VIEW_FACING
        last_cost_msg.action = Marker.ADD
        last_cost_msg.pose = Pose(position=Point(x=self.mppi.noisy_states[0, best_idx, -1, 0], y=self.mppi.noisy_states[0, best_idx, -1, 1]), orientation=Quaternion(w=1.0))
        last_cost_msg.scale = Vector3(x=2, y=2, z=2)
        last_cost_msg.color = ColorRGBA(r=1., g=1., b=1., a=1.)
        last_cost_msg.lifetime = rospy.Duration(0.5)
        last_cost_msg.frame_locked=True
#        last_cost_msg.text = "{:.2f}".format(self.mppi.costs[0, best_idx].item())
        last_cost_msg.text = "{:.2f}m/s".format(self.mppi.noisy_states[0, best_idx, :, 3].mean().item())
        marker_msg.markers.append(last_cost_msg)

        """
        if self.waypoint is not None:
            for i, goal_pose in enumerate(self.waypoint.poses[:2]):
                goal_msg = Marker()
                goal_msg.header.stamp = rospy.Time.now()
                goal_msg.header.frame_id = self.base_frame
                goal_msg.ns = "mppi"
                goal_msg.id = 3 + 2*i
                goal_msg.type = Marker.SPHERE
                goal_msg.action = Marker.ADD
                goal_msg.pose = Pose(position=Point(x=goal_pose.position.x, y=goal_pose.position.y), orientation=Quaternion(w=1.0))
                goal_msg.scale = Vector3(x=3.0, y=3.0, z=3.0)
                goal_msg.color = ColorRGBA(r=1., g=1., b=0., a=0.5)
                goal_msg.lifetime = rospy.Duration(0.5)
                goal_msg.frame_locked = True

                goal_txt_msg = Marker()
                goal_txt_msg.header.stamp = rospy.Time.now()
                goal_txt_msg.header.frame_id = self.base_frame
                goal_txt_msg.ns = "mppi"
                goal_txt_msg.id = 3 + 2*i
                goal_txt_msg.type = Marker.TEXT_VIEW_FACING
                goal_txt_msg.action = Marker.ADD
                goal_txt_msg.pose = Pose(position=Point(x=goal_pose.position.x, y=goal_pose.position.y), orientation=Quaternion(w=1.0))
                goal_txt_msg.scale = Vector3(x=3.0, y=3.0, z=3.0)
                goal_txt_msg.color = ColorRGBA(r=1., g=1., b=1., a=1.)
                goal_txt_msg.lifetime = rospy.Duration(0.5)
                goal_txt_msg.frame_locked = True
                goal_txt_msg.text = "mppi/goal_{}".format(i+1)

                marker_msg.markers.append(goal_msg)
                marker_msg.markers.append(goal_txt_msg)
        """

        """
        best_cost = self.mppi.costs.min()
        worst_cost = self.mppi.costs.max()
        for i, traj in enumerate(self.mppi.noisy_states[:16]):
            sample_msg = Marker()
            sample_msg.header.stamp = rospy.Time.now()
            sample_msg.header.frame_id = self.base_frame
            sample_msg.ns = "mppi"
            sample_msg.id = 2 + i
            sample_msg.type = Marker.LINE_STRIP
            sample_msg.action = Marker.ADD
            sample_msg.pose = Pose(orientation=Quaternion(w=1.0))
            sample_msg.scale = Vector3(x=0.05, y=0.05, z=0.05)
            c = (self.mppi.costs[i] - best_cost) / (worst_cost-best_cost)
            sample_msg.color = ColorRGBA(r=1-c, g=0., b=c, a=0.3)
            sample_msg.lifetime = rospy.Duration(0.5)
            sample_msg.frame_locked=True
            sample_msg.points = self.traj_to_pointarray(traj)
            marker_msg.markers.append(sample_msg)
        """

        self.viz_pub.publish(marker_msg)
        
    def traj_to_pointarray(self, traj):
        """
        Grab x, y from thr traj and make into pointarray
        """
        points = []
        colors = []
        smin = 1.
        smax = 5.

        for p in traj:
            msg = Point(x=p[0], y=p[1], z=0.)
            points.append(msg)

            z = (p[3] - smin) / (smax-smin)
            z = min(max(z, 0.), 1.)
            color_msg = ColorRGBA(r=z, g=0., b=1-z, a=1.)
            colors.append(color_msg)
        return points, colors

    def quat_to_yaw(self, orientation):
        qw = orientation.w
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
        return yaw

    def yaw_to_quat(self, yaw):
        quat = Quaternion()
        quat.x = 0.
        quat.y = 0.
        quat.z = (0.5 * yaw).sin().item()
        quat.w = (0.5 * yaw).cos().item()
        return quat

    def get_mppi_stats(self):
        best_idx = self.mppi.costs[0].argmin()
        traj = self.mppi.noisy_states[0, best_idx]
        cost = self.mppi.costs[0, best_idx]
        msg = MPPIStats()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame

        msg.average_speed = traj[:, 3].mean().item()
        msg.cost = cost.item()
        #not sure if really curvature, but summing up the diffs in theta
        msg.average_curvature = (traj[1:, 2] - traj[:-1, 2]).abs().sum().item()
        msg.cost_quantiles = torch.quantile(self.mppi.costs[0].cpu(), torch.linspace(0., 1., 100)).squeeze().numpy()

        for i,s in enumerate(traj):
            s_msg = SteerSetpointKBMState()
            s_msg.header.seq = i
            s_msg.header.stamp = msg.header.stamp + rospy.Duration(i * self.mppi.model.dt)
            s_msg.header.frame_id = self.base_frame
            s_msg.x = s[0].item()
            s_msg.y = s[1].item()
            s_msg.theta = s[2].item()
            s_msg.v = s[3].item()
            s_msg.delta = s[4].item()
            msg.trajectory.append(s_msg)

        return msg

    def spin(self):
        torch.set_printoptions(sci_mode=False)
        while not rospy.is_shutdown():
#            can_compute_costs = self.cost_fn.can_compute_cost()

            converter_can_get_data = self.converter.can_get_data()
            converter_status = self.converter.get_status_str()

            rospy.loginfo_throttle(1.0, 'ROS cvt status:\n{}'.format(converter_status))

            if converter_can_get_data:
                t1 = time.time()
                data = self.converter.get_data()

                #unsqueeze everything because mpc is batch
                data = dict_map(data, lambda x:x.unsqueeze(0))

                # load state
                states = self.model.get_observations(data).to(self.mppi.device).float()

                #load cost terms
                self.cost_fn.data = data

                t2 = time.time()

                for _ in range(self.mppi_itrs-1):
                    self.mppi.get_control(states, step=False)

                u, feasible = self.mppi.get_control(states)

                feasible = feasible.cpu()[0].item()

                if not feasible:
                    rospy.logwarn('No feasible trajectory found! Kill throttle')
                    u = u.cpu()[0]
                    u[0] = 0.
                else:
                    self.u_seq = self.mppi.last_controls[0].clone().cpu()
                    u = u.cpu()[0]

                self.u = self.alpha * self.u + (1.-self.alpha) * u

                #Get mppi stats
                best_idx = self.mppi.costs[0].argmin()
                mppi_stats = self.get_mppi_stats()
                self.stats_pub.publish(mppi_stats)

                t3 = time.time()

                rospy.loginfo_throttle(1.0, 'DATA LOAD: {:.4f}'.format(t2-t1))
                rospy.loginfo_throttle(1.0, 'MPPI: {:.4f}'.format(t3-t2))

                #debug
                rospy.loginfo_throttle(1.0, 'final speed = {:.2f}'.format(self.mppi.noisy_states[0, best_idx, -1, 3]))

                #For now I know the controls are [v, steer_setpoint]
                msg_out = Joy()
                msg_out.header.stamp = rospy.Time.now()
                msg_out.buttons = [0] * 12
                msg_out.axes = [0.] * 6
                msg_out.buttons[4] = 1
                msg_out.axes[2] = self.u[1].cpu().item() / 0.52 #/ self.model.u_ub[1] #normalize as max steer = 1 in joy
                self.cmd_pub.publish(msg_out)
                self.vel_pub.publish(Float32(data=self.u[0].cpu().item()))
            else:
                rospy.logwarn_throttle(1.0, "converter missing topics!")

                msg_out = Joy()
                msg_out.header.stamp = rospy.Time.now()
                msg_out.buttons = [0] * 12
                msg_out.axes = [0.] * 6
                msg_out.buttons[4] = 0
                msg_out.axes[2] = 0.
                self.cmd_pub.publish(msg_out)
                self.vel_pub.publish(Float32(0.))

            if self.use_actuator_delay:
                self.model.add_to_buffer(self.u)

            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('mppi_node')

    #data params
    config_fp = rospy.get_param("~config_fp")
    base_frame = rospy.get_param("~base_frame")
    mpc_itrs = rospy.get_param("~mpc_itrs")

    ##############################
    config = yaml.safe_load(open(config_fp, 'r'))
    mpc = setup_mpc(config)
    model = mpc.model
    cost_fn = mpc.cost_fn
    dt = config['common']['dt']
    config['ros']['dt'] = dt #make the ROS converter happy
    ##############################

    ros_cvt = ROSConverter(config['ros'])

    rate = rospy.Rate(1)
    rate.sleep()
    mppi_node.spin()
