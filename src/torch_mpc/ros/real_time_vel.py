#!/usr/bin/python3

import rospy
import numpy as np
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

global vel_pub
global br_static
global br
def odom_callback(msg):
    vel = Float32()
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    vz = msg.twist.twist.linear.z

    vel.data = np.linalg.norm([vx,vy,vz])

    vel_pub.publish(vel)



if __name__ == '__main__':
    rospy.init_node('real_time_vel')
    vel_pub = rospy.Publisher('/velocity', Float32, queue_size=10)
    rospy.Subscriber('/integrated_to_init', Odometry, odom_callback)
    rospy.spin()
