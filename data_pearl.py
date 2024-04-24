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

def ros_to_numpy(msg):
# assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())

    data_out = []

    origin = np.array([
        msg.info.pose.position.x - msg.info.length_x/2.,
        msg.info.pose.position.y - msg.info.length_y/2.
    ])

    map_width = np.array([msg.info.length_x])
    map_height = np.array([msg.info.length_y])

    res_x = []
    res_y = []

    for channel in range(1):  # just one channel
        # print(channel)
        idx = msg.layers.index('costmap')
        layer = msg.data[idx]
        height = layer.layout.dim[0].size
        width = layer.layout.dim[1].size
        data = np.array(list(layer.data), dtype=np.float32) #Why was hte data a tuple?
        data = data.reshape(height, width)

        data[~np.isfinite(data)] = None

        data = cv2.resize(data, dsize=(32, 32), interpolation=cv2.INTER_AREA)
        
        data_out.append(data[::-1, ::-1]) #gridmaps index from the other direction.
        res_x.append(msg.info.length_x / data.shape[0])
        res_y.append(msg.info.length_y / data.shape[1])

    data_out = np.stack(data_out, axis=0)

    reses = np.concatenate([np.stack(res_x), np.stack(res_y)])
    assert max(np.abs(reses - np.mean(reses))) < 1e-4, 'got inconsistent resolutions between gridmap dimensions/layers. Check that grid map layes are same shape and that size proportional to msg size'
    output_resolution = np.mean(reses, keepdims=True)

    metadata = {
        'origin': origin.tolist(),
        'resolution': output_resolution.item(),
        'width': map_width.item(),
        'height': map_height.item(),
        'feature_keys': 'costmap'
    }

    return {
            'data': data_out,
            'metadata': metadata
            }

if __name__ == '__main__':
    # YAML LOADING NOT WORKING CRYING
    
    # config_dir = "~/aec/torch_mpc/configs/costmap_speedmap.yaml"
    # with open(config_dir) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # data_output_dir = config['data_output_dir']

    print("welcome")
    state_topic = '/integrated_to_init'
    steer_topic = '/ros_talon/current_position'
    costmaps_topic = '/local_costmap'
    waypoints_topic = '/next_waypoints/odom'
    print("waypoints stuff")

    topics = [state_topic, steer_topic, costmaps_topic, waypoints_topic]

    run_dir = "../data/2024-04-17-14-31-20_to_warehouse_loop/2024-04-17-14-31-20_0.bag"

    curr_s = 0.
    prev_pos = None
        
        # for bfp in bfps:
    bag = rosbag.Bag(run_dir, 'r')
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == state_topic:
            print('state')
            # x = msg.pose.pose.position.x
            # y = msg.pose.pose.position.y

        elif topic == steer_topic:
            # steer = msg.data
            print('steer')
        elif topic == costmaps_topic:
            print('costmaps')
            # print(msg.layers.index('costmap'))
            costmaps = ros_to_numpy(msg)
            print(f"Resolution: {costmaps['metadata']['resolution']}")
            print(f"Origin: {costmaps['metadata']['origin']}")
            print(f"Width: {costmaps['metadata']['width']}")
            print(f"Height: {costmaps['metadata']['height']}")
            print(f"Feature Keys: {costmaps['metadata']['feature_keys']}")
            #print(f"Data: {costmaps['data']}")
        elif topic == waypoints_topic:
            print("waypoints")
            # print(msg)
            # waypoints = msg.data