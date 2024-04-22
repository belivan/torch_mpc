import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

data_dir = '/home/anton/Desktop/SPRING24/AEC/torch_mpc/src/torch_mpc/algos/aglos_data'
X_jit = torch.jit.load(os.path.join(data_dir, 'X.pt'))
U_jit = torch.jit.load(os.path.join(data_dir, 'U.pt'))
traj_jit = torch.jit.load(os.path.join(data_dir, 'traj.pt'))

X = list(X_jit.parameters())[0].cpu().numpy()
U = list(U_jit.parameters())[0].cpu().numpy()
traj = list(traj_jit.parameters())[0].cpu().numpy()

config_fp = '/home/atv/physics_atv_ws/src/control/torch_mpc/configs/test_config.yaml'
config = yaml.safe_load(open(config_fp, 'r'))
batch_size = config['common']['B']
device = config['common']['device']

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].set_title("Traj")
for b in range(batch_size):
    axs[0].plot(X[b, :, 0], X[b, :, 1], c='b', label='traj' if b == 0 else None)
    axs[0].plot(traj[b, :, 0], traj[b, :, 1], c='r', label='pred' if b == 0 else None)
    axs[1].plot(U[b, :, 0], c='r', label='throttle' if b == 0 else None)
    axs[1].plot(U[b, :, 1], c='b', label='steer' if b == 0 else None)

axs[1].set_title("Controls")
axs[2].set_title("States")

colors = 'rgbcmyk'
for xi, name in enumerate(['X', 'Y', 'Th', 'V', 'W']):
    c = colors[xi % len(colors)]
    for b in range(batch_size):
        axs[2].plot(X[b, :, xi], c=c, label=name if b == 0 else None)

axs[0].legend()
axs[1].legend()
axs[2].legend()

axs[0].set_aspect('equal')
plt.show()
