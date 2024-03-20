import yaml
import rospy
import torch
import numpy as np

from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from grid_map_msgs.msg import GridMap
from torch_state_lattice_planner.msg import ValueMapHeading

class Float32Convert:
    def __init__(self):
        self.msg_type = Float32

    def cvt(self, msg):
        return torch.tensor([msg.data])

class OdometryConvert:
    def __init__(self):
        self.msg_type = Odometry

    def cvt(self, msg):
        return torch.tensor([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])

class PoseArrayConvert:
    def __init__(self, goals=True):
        self.msg_type = PoseArray
        self.goals = goals

    def cvt(self, msg):
        res = []
        for pose in msg.poses:
            x = torch.tensor([
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ])
            res.append(x[:2] if self.goals else x)

        return torch.stack(res, dim=0)

class GridMapConvert:
    def __init__(self, key):
        self.msg_type = GridMap
        self.key = key

    def cvt(self, msg):
        metadata = {
            'origin': torch.tensor([
                msg.info.pose.position.x - 0.5*msg.info.length_x,
                msg.info.pose.position.y - 0.5*msg.info.length_y
            ]),
            'length_x': torch.tensor(msg.info.length_x),
            'length_y': torch.tensor(msg.info.length_y),
            'resolution':  torch.tensor(msg.info.resolution)
        }
        nx = round(msg.info.length_x/msg.info.resolution)
        ny = round(msg.info.length_y/msg.info.resolution)

        idx = msg.layers.index(self.key)
        data = np.array(msg.data[idx].data).reshape(nx, ny)[::-1, ::-1].copy()
        data = torch.tensor(data)

        return {
            'data': data,
            'metadata': metadata
        }

class ValueMapHeadingConvert:
    def __init__(self):
        self.msg_type = ValueMapHeading

    def cvt(self, msg):
        metadata = {
            'origin': torch.tensor([
                msg.info.pose.position.x - 0.5*msg.info.length_x,
                msg.info.pose.position.y - 0.5*msg.info.length_y
            ]),
            'length_x': torch.tensor(msg.info.length_x),
            'length_y': torch.tensor(msg.info.length_y),
            'resolution':  torch.tensor(msg.info.resolution),
            'headings': torch.tensor(msg.headings),
        }
        nx = round(msg.info.length_x/msg.info.resolution)
        ny = round(msg.info.length_y/msg.info.resolution)
        nh = len(msg.headings)

        data = torch.tensor(np.array(msg.data.data).reshape(nx, ny, nh))

        return {
            'data': data,
            'metadata': metadata
        }

str_to_cvt_class = {
    'Float32': Float32Convert,
    'Odometry': OdometryConvert,
    'PoseArray': PoseArrayConvert,
    'GridMap': GridMapConvert,
    'ValueMapHeading': ValueMapHeadingConvert
}

def pp_torch_dict(x, prefix=''):
    for k,v in x.items():
        if isinstance(v, dict):
            print('{}{}:'.format(prefix, k))
            pp_torch_dict(v, prefix+'  ')
        else:
            print('{}{}:{}'.format(prefix, k, v.shape))

## TODO move above into own files

class ROSConverter:
    """
    Class that handles all the ROS subscribing for MPC
    """
    def __init__(self, config):
        self.config = config
        self.subscribers = {}
        self.converters = {}

        self.data = {}
        self.data_times = {}

        self.setup_subscribers()

    def setup_subscribers(self):
        for topic_conf in self.config['topics']:
            self.data[topic_conf['name']] = None
            self.data_times[topic_conf['name']] = -1.
            self.converters[topic_conf['name']] = str_to_cvt_class[topic_conf['type']](**topic_conf['args'])

            sub = rospy.Subscriber(topic_conf['topic'], self.converters[topic_conf['name']].msg_type, self.handle_msg, callback_args=topic_conf)

    def handle_msg(self, msg, topic_conf):
            self.data[topic_conf['name']] = msg
            self.data_times[topic_conf['name']] = rospy.Time.now().to_sec()

    def get_data(self, device='cpu'):
        return {k:self.converters[k].cvt(msg) for k,msg in self.data.items()}

    def can_get_data(self):
        return all([rospy.Time.now().to_sec() - dt < self.config['max_age'] for dt in self.data_times.values()])

    def get_status_str(self):
        out = '---converter status--- \n'
        for topic_conf in self.config['topics']:
            data_exists = self.data[topic_conf['name']] is not None
            data_age = rospy.Time.now().to_sec() - self.data_times[topic_conf['name']]
            out += '\t{:<16} exists: {} age:{:.2f}s\n'.format(topic_conf['name']+':', data_exists, data_age)

        out += 'can get data: {}'.format(self.can_get_data())
        out += '\n'
        return out

if __name__ == '__main__':
    import time
    config_fp = '/home/physics_atv/physics_atv_ws/src/control/torch_mpc/configs/debug.yaml'
    config = yaml.safe_load(open(config_fp, 'r'))['ros']

    rospy.init_node('torch_mpc_ros_converter')

    converter = ROSConverter(config)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        print(converter.get_status_str())

        if converter.can_get_data():
            t1 = time.time()
            data = converter.get_data()
            t2 = time.time()

            rospy.loginfo('cvt time: {:.4f}s'.format(t2-t1))
            pp_torch_dict(data)

        rate.sleep()
