import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

# Set current directory to this file's directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

traj_jit = torch.jit.load('algos_data/TRAJ.pt')
U_jit = torch.jit.load('algos_data/U.pt')
X_jit = torch.jit.load('algos_data/X.pt')
X_real_jit = torch.jit.load('algos_data/X_TRUE.pt')

X = list(X_jit.parameters())[0].cpu().numpy()
U = list(U_jit.parameters())[0].cpu().numpy()
traj = list(traj_jit.parameters())[0].cpu().numpy()
X_real = list(X_real_jit.parameters())[0].cpu().numpy()

print(X.shape)
print(U.shape)
print(traj.shape)
print(X_real.shape)

X = np.concatenate(X, axis=1)
U = np.concatenate(U, axis=1)
traj = np.concatenate(traj, axis=1)
X_real = np.concatenate(X_real, axis=1)

# Find config file relative to this file
config_fp = '../../../configs/costmap_speedmap.yaml'
# Load config file
config = yaml.safe_load(open(config_fp, 'r'))
batch_size = config['common']['B']
device = config['common']['device']

num_trajectories = X.shape[0]

# for i in range(num_trajectories):
#     plt.plot(X[i, :, 0], X[i, :, 1], label=f'Trajectory {i}')
#     plt.plot(traj[i, :, 0], traj[i, :, 1], '--', label=f'Prediction {i}')
# plt.legend()
# plt.title("Overlay of Trajectories")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
for i in range(num_trajectories):
    if i == 0:
        plt.plot(X[i, :, 0], X[i, :, 1], 'b-', label='Trajectory')
        plt.plot(traj[i, :, 0], traj[i, :, 1], 'r--', label='Prediction (KBM)')
    else:
        plt.plot(X[i, :, 0], X[i, :, 1], 'b-')
        plt.plot(traj[i, :, 0], traj[i, :, 1], 'r--')
plt.plot(X_real[:, 0], X_real[:, 1], 'g', label='GT')
plt.legend()
plt.title("Results Comparison")
plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.savefig('algos_data/visualize_all_comapre_no_goals.png')


# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# axs[0].set_title("Traj")
# for b in range(batch_size):
#     axs[0].plot(X[b, :, 0], X[b, :, 1], c='b', label='traj' if b == 0 else None)
#     axs[0].plot(traj[b, :, 0], traj[b, :, 1], c='r', label='pred' if b == 0 else None)
#     axs[1].plot(U[b, :, 0], c='r', label='throttle' if b == 0 else None)
#     axs[1].plot(U[b, :, 1], c='b', label='steer' if b == 0 else None)

# axs[1].set_title("Controls")
# axs[2].set_title("States")

# colors = 'rgbcmyk'
# for xi, name in enumerate(['X', 'Y', 'Th']):
#     c = colors[xi % len(colors)]
#     for b in range(batch_size):
#         axs[2].plot(X[b, :, xi], c=c, label=name if b == 0 else None)

# axs[0].legend()
# axs[1].legend()
# axs[2].legend()

# # axs[0].set_aspect('equal')
# # plt.show()
# plt.savefig('algos_data/visualize_samples.png')
