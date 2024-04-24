import os
import yaml
import rosbag
import argparse
import numpy as np
import cv2

def quat_to_yaw(quat):
    """
    Convert a quaternion (as [x, y, z, w]) to yaw
    """
    return np.arctan2(2 * (quat[3]*quat[2] + quat[0]*quat[1]), 1 - 2 * (quat[1]**2 + quat[2]**2))

if __name__ == 'main':
    # open config file
    # config_dir = "aec/torch_mpc/configs/costmap_speedmap.yaml"
    # with open('config.yaml') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # data_output_dir = config['data_output_dir']
    print("welcome")
    state_topic = '/integrated_to_init'
    steer_topic = '/ros_talon/current_position'
    costmaps_topic = '/local_costmap'
    waypoints_topic = '/next_waypoints/odom'
    print("waypoints stuff")
    
    topics = [state_topic, steer_topic, costmaps_topic, waypoints_topic]
    
    run_dir = "~/aec/data/2024-04-17-14-31-20_1.bag"
    # bfps = sorted([x for x in os.listdir(run_dir) if x[-4:] == '.bag'])
    
    curr_s = 0.
    prev_pos = None
    
    # for bfp in bfps:
    bag = rosbag.Bag(run_dir, 'r')
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == state_topic:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y

        elif topic == steer_topic:
            # steer = msg.data
            print('steer')
        elif topic == costmaps_topic:
            length_x = msg.info.length_x
            length_y = msg.info.length_y
            map_width = np.array([msg.info.length_x])
            map_height = np.array([msg.info.length_y])
            origin = np.array([
                msg.info.pose.position.x - length_x/2.,
                msg.info.pose.position.y - length_y/2.
                                ])
            print("origin: ", origin)
        elif topic == waypoints_topic:
            waypoints = msg.data
