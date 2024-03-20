import rospy
import rosbag
import numpy as np
import os
import argparse
import time

from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import Odometry

def get_waypoints(bag_dir, waypoint_spacing, odom_topic):
    cnt = 0.

    fps = os.listdir(bag_dir)
    fps.sort(key=lambda x: os.path.getmtime(os.path.join(bag_dir, x)))
    print(fps)

    wpts = []
    last_pos = None
    base_frame = None

    for fp in fps:
        bag = rosbag.Bag(os.path.join(bag_dir, fp), 'r')
        for topic, msg, t in bag.read_messages(topics=[odom_topic]):
            pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

            if last_pos is not None:
                cnt += np.linalg.norm(pos - last_pos)

            if cnt > waypoint_spacing:
                wpts.append(pos)
                cnt = 0.

            last_pos = pos
            base_frame = msg.header.frame_id

    #add the last wpt too
    wpts.append(last_pos)
    wpts = np.stack(wpts, axis=0)
    return wpts, base_frame

curr_pos = None
def handle_odom(msg):
    global curr_pos
    curr_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', type=str, required=True, help='dir of bags to extract waypoints from')
    parser.add_argument('--waypoint_spacing', type=float, required=False, default=20., help='distance to sapce waypoints')
    parser.add_argument('--odom_topic', type=str, required=False, default='/nav/relative_pos/odom', help='topic to get odom from')
    args = parser.parse_args()
    
    wpts, base_frame = get_waypoints(args.bag_dir, args.waypoint_spacing, args.odom_topic)

    rospy.init_node('waypoint_publisher')
    rate = rospy.Rate(5)

    pub = rospy.Publisher('/next_waypoints/odom', PoseArray, queue_size=1)
    sub = rospy.Subscriber(args.odom_topic, Odometry, handle_odom, queue_size=1)

    while not rospy.is_shutdown():
        print(curr_pos, wpts[0], end='\r')
        if curr_pos is not None and np.linalg.norm(curr_pos - wpts[0]) < 2. and len(wpts) > 1:
            wpts = wpts[1:]

        wpt_msg = PoseArray()
        wpt_msg.header.frame_id = base_frame
        for wpt in wpts:
            pose_msg = Pose()
            pose_msg.position.x = wpt[0]
            pose_msg.position.y = wpt[1]
            pose_msg.orientation.w = 1.
            wpt_msg.poses.append(pose_msg)

        wpt_msg.header.stamp = rospy.Time.now()
        pub.publish(wpt_msg)
        rate.sleep()
