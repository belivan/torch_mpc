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

terminal_traj = 0
terminal_traj_last = 0

while terminal_traj < num_trajectories - 1:
    print("GRAPHING")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # for i in range(terminal_traj_last, terminal_traj):
        if i == terminal_traj_last:
            axs[0].plot(X[i, :, 0], X[i, :, 1], 'b-', label='Trajectory', alpha=0.1)

            axs[1].plot(U[i, :, 0], 'r-', label='Throttle')
            axs[1].plot(U[i, :, 1], 'b-', label='Steer')

            # axs[0].scatter(goals[i, 0, 0, 0], goals[i, 0, 0, 1], c='g', label='Goal 1')
            # axs[0].scatter(goals[i, 1, 0, 0], goals[i, 1, 0, 1], c='r', label='Goal 2')
        if i > terminal_traj_last:
            if (goals[i-1, 0, 0, 0] != goals[i, 0, 0, 0] and goals[i-1, 0, 0, 1] != goals[i, 0, 0, 1]):
                terminal_traj = i
                print("FOUND TERMINAL TRAJ at Goal 1")
                print("Goal 1: ", goals[i, 0, 0, 0], goals[i, 0, 0, 1])
                print("Sampled Traj: ", i)
                break
            elif (goals[i-1, 1, 0, 0] != goals[i, 1, 0, 0] and goals[i-1, 1, 1, 1] != goals[i, 1, 0, 1]):
                terminal_traj = i
                print("FOUND TERMINAL TRAJ at Goal 2")
                print("Goal 2: ", goals[i, 1, 0, 0], goals[i, 1, 0, 1])
                print("Sampled Traj: ", i)
                break

        # if i % 50 == 0:
        axs[0].plot(X[i, :, 0], X[i, :, 1], 'b-', alpha=0.1)
        axs[1].plot(U[i, :, 0], 'r-')
        axs[1].plot(U[i, :, 1], 'b-')

        terminal_traj = i

    axs[0].plot(X_real[terminal_traj-10:terminal_traj, 0], X_real[terminal_traj-10:terminal_traj, 1], 'g', label='GT')
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
        for i in range(terminal_traj-10, terminal_traj):
            if i == 0:
                axs[2].plot(np.arange(i, i+length, 1), X[i, :, xi], c=c, label=name)
                axs[2].plot(X_real[:, xi], label=name + ' GT', alpha=0.1)
            if i % 50 == 0:
                axs[2].plot(np.arange(i, i+length, 1), X[i, :, xi], c=c)
            axs[2].plot(X_real[:, xi], alpha=0.1)
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("State Value")
    axs[2].legend()
    # plt.title("Trajectories and Ground Truth States")

    # plt.savefig('algos_data/visualize_2000_comapre_ALL_no_goals_H50_spaced_control.png')
    plt.show(block=False)
    plt.pause(0.2)
    plt.clf()

    print("DONE")

    terminal_traj_last = terminal_traj
    print("Terminal Traj: ", terminal_traj)
