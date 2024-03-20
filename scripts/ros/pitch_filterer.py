#! /usr/bin/python3

import rospy
import numpy as np

from scipy.spatial.transform import Rotation

from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

class PitchFilterer:
    """
    Simple node that low-passes and publishes the vehicle pitch
    """
    def __init__(self, odom_topic, pitch_topic, n):
        self.pitch_buffer = np.zeros(n)

        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)
        self.pitch_pub = rospy.Publisher(pitch_topic, Float32, queue_size=1)

    def handle_odom(self, msg):
        q = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        pitch = Rotation.from_quat(q).as_euler('xyz')[1]
        self.pitch_pub.publish(Float32(pitch))

if __name__ == '__main__':
    rospy.init_node('pitch_filterer')

    odom_topic = rospy.get_param('~odom_topic')
    pitch_topic = rospy.get_param('~pitch_topic')
    n = rospy.get_param('~N', 10)

    filterer = PitchFilterer(odom_topic, pitch_topic, n)
    rospy.spin()
