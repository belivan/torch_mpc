import torch
import numpy as np
import os
import matplotlib.pyplot as plt

import yaml

# Set current directory to this file's directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))
data_path = 'algos_data/test1/'
traj_jit = torch.jit.load(data_path + 'TRAJ.pt')
U_jit = torch.jit.load(data_path + 'U.pt')
X_jit = torch.jit.load(data_path + 'X.pt')
X_real_jit = torch.jit.load(data_path + 'X_TRUE.pt')
goals_jit = torch.jit.load(data_path + 'GOALS.pt')

X = list(X_jit.parameters())[0].cpu().numpy()
U = list(U_jit.parameters())[0].cpu().numpy()
traj = list(traj_jit.parameters())[0].cpu().numpy()
X_real = list(X_real_jit.parameters())[0].cpu().numpy()
goals = list(goals_jit.parameters())[0].cpu().numpy()

print(X.shape)
print(U.shape)
print(traj.shape)
print(X_real.shape)
print(goals.shape)

X = np.concatenate(X, axis=1)
U = np.concatenate(U, axis=1)
traj = np.concatenate(traj, axis=1)
X_real = np.concatenate(X_real, axis=1)

# U_mean = np.mean(U, axis=1)

print(X.shape)
print(U.shape)

# Find config file relative to this file
config_fp = '../../../configs/costmap_speedmap.yaml'
# Load config file
config = yaml.safe_load(open(config_fp, 'r'))
batch_size = config['common']['B']
length = config['common']['H']
device = config['common']['device']

num_trajectories = X.shape[0]

current_traj = 0
# prev_traj = 0

while current_traj < num_trajectories - 1:
    print("GRAPHING")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].plot(X[current_traj, :, 0], X[current_traj, :, 1], 'b-', label='Trajectory', alpha=0.1)

    axs[1].plot(U[current_traj, :, 0], 'r-', label='Throttle')
    axs[1].plot(U[current_traj, :, 1], 'b-', label='Steer')

    # axs[0].scatter(goals[i, 0, 0, 0], goals[i, 0, 0, 1], c='g', label='Goal 1')
    # axs[0].scatter(goals[i, 1, 0, 0], goals[i, 1, 0, 1], c='r', label='Goal 2')

    dx_total = (goals[current_traj, 0, 0, 0] - X[current_traj, 0, 0])
    dy_total = (goals[current_traj, 0, 0, 1] - X[current_traj, 0, 1])

    dx = dx_total / np.sqrt(dx_total**2 + dy_total**2)
    dy = dy_total / np.sqrt(dx_total**2 + dy_total**2)

    axs[0].arrow(X[current_traj, 0, 0], X[current_traj, 0, 1], dx, dy,
          shape='full', lw=0.2, length_includes_head=True, head_width=0.02, color='black', label='Goal 1')

    if current_traj < 50:
        index = 0
    else:
        index = current_traj - 50

    axs[0].plot(X_real[index:current_traj, 0], X_real[index:current_traj, 1], 'g', label='GT')
    axs[0].set_xlabel("X Position")
    axs[0].set_ylabel("Y Position")
    axs[0].legend()
    # plt.title("Trajectories and Ground Truth Position")

    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Control Value")
    axs[1].legend()
    # plt.title("Trajectories and Controls")

    colors = 'rgbcmyk'
    for xi, name in enumerate(['X', 'Y', 'Th']):
        c = colors[xi % len(colors)]
        axs[2].plot(np.arange(current_traj, current_traj + length, 1), X[current_traj, :, xi], c=c, label=name)
        axs[2].plot(X_real[index:current_traj, xi], label=name + ' GT', alpha=0.1)
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("State Value")
    axs[2].legend()
    # plt.title("Trajectories and Ground Truth States")

    # plt.savefig('algos_data/visualize_2000_comapre_ALL_no_goals_H50_spaced_control.png')
    manager = plt.get_current_fig_manager()
    manager.window.attributes('-fullscreen', True)

    plt.show(block=False)
    # plt.show()
    plt.pause(0.07)
    plt.clf()

    print("DONE")

    current_traj += 10
    print("Current Trajectory", current_traj)
