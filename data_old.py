import os
import yaml
import rosbag
import argparse
import numpy as np
import cv2
import torch

prev_pos = None
curr_s = 0.

def quat_to_yaw(quat):
    """
    Convert a quaternion (as [x, y, z, w]) to yaw
    """
    return np.arctan2(2 * (quat[3]*quat[2] + quat[0]*quat[1]), 1 - 2 * (quat[1]**2 + quat[2]**2))
def waypoint_to_numpy(msg, waypoints):
    p = np.array([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z])

    q = np.array([
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w])

    if prev_pos is None:
        prev_pos = p
        return None

    ds = np.linalg.norm(prev_pos[:2] - p[:2])
    if ds > 50.:
        print('large gps jump. skipping...') 
        return None

    curr_s += ds
    prev_pos = np.concatenate([p, q])

    # add a waypoint
    if curr_s > 25:
        yaw = quat_to_yaw(q)
        wpt = np.concatenate([p, yaw.reshape(1)])
        waypoints.append(wpt)

        curr_s = 0.

    # always make final pose a waypoint
    if curr_s > 0.2 * 25:
        yaw = quat_to_yaw(prev_pos[3:7])
        wpt = np.concatenate([prev_pos[:3], yaw.reshape(1)])
        waypoints.append(wpt)

    waypoints = np.stack(waypoints, axis=0)
    
    #convert to yaml
    res = []
    for wpt in waypoints:
        wdict = {
            'frame_id': frame_id,
            'pose': {
                'x': wpt[0].item(),
                'y': wpt[1].item(),
                'z': wpt[2].item(),
                'yaw': wpt[3].item()
            },
            'radius': 4.0
        }
        res.append(wdict)
    return res

def costmap_to_numpy(msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())

    data_out = []

    origin = np.array([
        msg.info.pose.position.x - msg.info.length_x/2.,
        msg.info.pose.position.y - msg.info.length_y/2.
    ])

    map_width = np.array([msg.info.length_x])
    map_height = np.array([msg.info.length_y])

    res_x = []
    res_y = []

    for channel in range(1):
        idx = msg.layers.index('costmap')
        layer = msg.data[idx]
        height = layer.layout.dim[0].size
        width = layer.layout.dim[1].size
        data = np.array(list(layer.data), dtype=np.float32) #Why was hte data a tuple?
        data = data.reshape(height, width)

        # fill with 99th percentile value of data
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
    #print('hello')
    # open config file
    config_dir = "configs/costmap_speedmap.yaml"
    with open(config_dir, 'r') as stream:
        config = yaml.safe_load(stream)

    data_output_dir = config['data_output_dir']
    state_topic = '/integrated_to_init'
    steer_topic = '/ros_talon/current_position'
    costmaps_topic = '/local_costmap'
    goals_topic = '/next_waypoints/odom'
    
    topics = [state_topic, steer_topic, costmaps_topic, goals_topic]
    
    run_dir = "../data/2024-04-17-14-31-20_1.bag"
    # bfps = sorted([x for x in os.listdir(run_dir) if x[-4:] == '.bag'])

    waypoints = []
    frame_id = None
    bag = rosbag.Bag(run_dir, 'r')
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == state_topic:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            print(f"Position: ({x}, {y})")

        elif topic == steer_topic:
            # steer = msg.data
            print('steer')
        elif topic == costmaps_topic:
            print('costmaps')
            # print(msg.layers.index('costmap'))
            costmaps = costmap_to_numpy(msg)
            print(f"Resolution: {costmaps['metadata']['resolution']}")
            print(f"Origin: {costmaps['metadata']['origin']}")
            print(f"Width: {costmaps['metadata']['width']}")
            print(f"Height: {costmaps['metadata']['height']}")
            print(f"Feature Keys: {costmaps['metadata']['feature_keys']}")
            print(f"Data: {costmaps['data']}")
        elif topic == goals_topic:
            print("waypoints")
            goals = waypoint_to_numpy(msg, waypoints)
            print(msg)
        
        res = {'waypoints':res, 'version':1.0}
        yaml.dump(res, open(os.path.join(args.save_to, 'forward.yaml'), 'w'))
        res['waypoints'] = list(reversed(res['waypoints']))
        yaml.dump(res, open(os.path.join(args.save_to, 'backward.yaml'), 'w'))
            