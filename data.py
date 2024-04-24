import os
import yaml
import rosbag
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torch import nn

class TensorContainer(nn.Module):
    def __init__(self, tensor_dict):
        super().__init__()
        for key, value in tensor_dict.items():
            setattr(self, key, value)

def quat_to_yaw(quat):
    """
    Convert a quaternion (as [x, y, z, w]) to yaw
    """
    return np.arctan2(2 * (quat[3]*quat[2] + quat[0]*quat[1]), 1 - 2 * (quat[1]**2 + quat[2]**2))  

# TOPICS:
state_topic = '/integrated_to_init'
steer_topic = '/ros_talon/current_position'
costmaps_topic = '/local_costmap'
goals_topic = '/next_waypoints/odom'

save_to = '../data/mppi_inputs/run_3/'  # make sure to change when needed
os.makedirs(save_to, exist_ok=True)  # output directory
run_dir = "../data/"  # 2024-04-17-14-31-20_1.bag

bfps = sorted([x for x in os.listdir(run_dir) if x[-4:] == '.bag'])

curr_s = 0.
prev_pos = None

waypoints_list = []
costmap_metadata = []
frame_id = None

pos_list = []
steer_list = []
data_list = []

for bfp in bfps:
    bag = rosbag.Bag(os.path.join(run_dir, bfp), 'r')
    for topic, msg, t in bag.read_messages(topics=[state_topic, 
                                                   steer_topic, 
                                                   costmaps_topic, 
                                                   goals_topic]):
        print(f"T: {t}")
        if topic == state_topic:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            # create a postion pair and save it as torch tensor
            pos = np.array([x, y])
            pos = torch.tensor(pos)
            pos = pos.unsqueeze(0)
            pos_list.append(pos)

        elif topic == steer_topic:
            steer = msg.data
            steer = np.array([steer])
            steer = torch.tensor(steer)
            steer = steer.unsqueeze(0)
            steer_list.append(steer)

        elif topic == costmaps_topic:
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
                data = np.array(list(layer.data), dtype=np.float32) # Why was hte data a tuple?
                data = data.reshape(height, width)

                # if no data, fill with 99th percentile value of data
                data[~np.isfinite(data)] = None

                data = cv2.resize(data, dsize=(32, 32), interpolation=cv2.INTER_AREA)

                data_out.append(data[::-1, ::-1]) # gridmaps index from the other direction.
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

            # Save the costmap data and metadata
            costmap = torch.tensor(data_out).unsqueeze(0)
            # print("COSTMAP SHAPE",costmap.shape)
            data_list.append(costmap)

            costmap_metadata.append(metadata)

        elif topic == goals_topic:
            if frame_id is None:
                frame_id = msg.header.frame_id

            p = np.zeros((len(msg.poses), 3))
            q = np.zeros((len(msg.poses), 4))  # If orientation data is needed

            # Extract positions and orientations
            for i, pose in enumerate(msg.poses):
                p[i] = [pose.position.x, pose.position.y, pose.position.z]
                q[i] = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

            if prev_pos is None:
                prev_pos = np.concatenate([p, q], axis=1)  # Initialize prev_pos with the positions of the first timestep
                continue

            # # Calculate distances between consecutive points in the list
            ds = np.linalg.norm(prev_pos[:, :2] - p[:, :2])

            # # Handling large GPS jumps
            # if ds > 50:
            #     print('Large GPS jump detected, skipping...')
            #     continue

            curr_s += np.sum(ds) / len(p)
            # prev_pos = np.concatenate([p[0], q[0]])

            # Check and append new waypoints
            if curr_s > 25:
                for i in range(len(p)):
                    yaw = quat_to_yaw(q[i])
                    wpt = np.concatenate([p[i], yaw.reshape(1)])
                    waypoints_list.append(wpt)
                curr_s = 0.

print("DONE SAMPLING")
print(f"Number of positions: {len(pos_list)}")
print(f"Number of costmaps: {len(data_list)}")
print(f"Number of waypoints: {len(waypoints_list)}")

# always make final pose a waypoint
if curr_s > 0.2 * 25:
    yaw = quat_to_yaw(prev_pos[-1, 3:7])
    wpt = np.concatenate([prev_pos[-1, :3], yaw.reshape(1)])
    waypoints_list.append(wpt)

waypoints = np.stack(waypoints_list, axis=0)

# if args.viz:
#     plt.scatter(waypoints[:, 0], waypoints[:, 1])
#     plt.gca().set_aspect(1.)
#     plt.

# convert to yaml
cmap = []
for md in costmap_metadata:
    cmap.append({
        'origin': md['origin'],
        'resolution': md['resolution'],
        'width': md['width'],
        'height': md['height'],
        'feature_keys': md['feature_keys']
    })

cmap = {'costmaps': cmap, 'version': 1.0}
os.makedirs(os.path.join(save_to, 'costmaps'), exist_ok=True)
yaml.dump(cmap, open(os.path.join(save_to, 'costmaps', 'metadata.yaml'), 'w'))
print("SAVED COSTMAP METADATA")

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

# Finally, save the waypoints
res = {'waypoints': res, 'version': 1.0}
os.makedirs(os.path.join(save_to, 'waypoints'), exist_ok=True)
yaml.dump(res, open(os.path.join(save_to, 'waypoints','forward.yaml'), 'w'))
res['waypoints'] = list(reversed(res['waypoints']))
yaml.dump(res, open(os.path.join(save_to, 'backward.yaml'), 'w'))
print("SAVED WAYPOINTS")


# Save the position and steering data
pos = torch.cat(pos_list, dim=0)
steer = torch.cat(steer_list, dim=0)
data = torch.cat(data_list, dim=0)

pos_path = os.path.join(save_to, 'pos', 'pos_data.pth')
os.makedirs(os.path.join(save_to, 'pos'), exist_ok=True)
steer_path = os.path.join(save_to, 'steer', 'steer_data.pth')
os.makedirs(os.path.join(save_to, 'steer'), exist_ok=True)
data_path = os.path.join(save_to, 'data', 'data.pth')
os.makedirs(os.path.join(save_to, 'data'), exist_ok=True)


# create pos, steer, data samples for one time step
tensor_dict = {'pos': pos[0],
               'steer': steer[0],
               'data': data[0],
               'waypoints': torch.tensor(waypoints[0])}

print(pos[0].shape)
print(steer[0].shape)
print(data[0].shape)
print(waypoints[0].shape)

tensors = TensorContainer(tensor_dict)
tensors = torch.jit.script(tensors)
sample_dir = os.path.join(save_to, 'sample', 'sample.pth')
os.makedirs(os.path.join(save_to, 'sample'), exist_ok=True)
tensors.save(sample_dir)
print("SAVED SAMPLE")

tensor_dict = {'data': pos}
tensors = TensorContainer(tensor_dict)
tensors = torch.jit.script(tensors)
tensors.save(pos_path)

tensor_dict = {'data': steer}
tensors = TensorContainer(tensor_dict)
tensors = torch.jit.script(tensors)
tensors.save(steer_path)

tensor_dict = {'data': data}
tensors = TensorContainer(tensor_dict)
tensors = torch.jit.script(tensors)
tensors.save(data_path)
print("SAVED DATA")
