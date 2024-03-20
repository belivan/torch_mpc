#!/usr/bin/python3

import rospy
import torch
import numpy as np
import time
import tf2_ros

import yaml

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float32
from geometry_msgs.msg import Pose, Point, Vector3,Quaternion, Vector3Stamped, PoseArray
from sensor_msgs.msg import Joy
from nav_msgs.msg import OccupancyGrid, Odometry
from copy import deepcopy
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from tf.transformations import euler_from_quaternion, quaternion_from_euler

global odom_pub
global br_static
global br
def odom_callback(msg):
    odom = Odometry()
    odom = deepcopy(msg)
    odom.pose.pose.orientation.x = msg.pose.pose.orientation.y
    odom.pose.pose.orientation.y = -msg.pose.pose.orientation.x

    odom.twist.twist.linear.x = msg.twist.twist.linear.y
    odom.twist.twist.linear.y = -msg.twist.twist.linear.x
    odom.twist.twist.angular.x = msg.twist.twist.angular.y
    odom.twist.twist.angular.y = -msg.twist.twist.angular.x

    odom_pub.publish(odom)

def tf_callback(msg):
    if msg.transforms[0].header.frame_id =='sensor_init' and msg.transforms[0].child_frame_id == 'vehicle':
        return
    else:
        for i in range(len(msg.transforms)):
            br.sendTransform(msg.transforms[i])

def tf_static_callback(msg):
    if msg.transforms[0].header.frame_id =='map' and msg.transforms[0].child_frame_id == 'sensor_init':
        return
    else:
        for i in range(len(msg.transforms)):
            br_static.sendTransform(msg.transforms[i])

if __name__ == '__main__':
    rospy.init_node('odom_tf')
    odom_pub = rospy.Publisher('/odometry/filtered_odom', Odometry, queue_size=10)
    br_static = tf2_ros.StaticTransformBroadcaster()
    br = tf2_ros.TransformBroadcaster()
    rospy.Subscriber('/old_odom', Odometry, odom_callback)
    rospy.Subscriber('/old_tf_static', TFMessage, tf_static_callback)
    rospy.Subscriber('/old_tf', TFMessage, tf_callback)
    rospy.spin()
