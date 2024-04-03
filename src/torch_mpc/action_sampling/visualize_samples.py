import torch
import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = '/home/anton/Desktop/SPRING24/AEC/torch_mpc/src/torch_mpc/action_sampling/sampling_data'
u_nominal_jit = torch.jit.load(os.path.join(data_dir, 'u_nominal.pt'))
u_lb_jit = torch.jit.load(os.path.join(data_dir, 'u_lb.pt'))
u_ub_jit = torch.jit.load(os.path.join(data_dir, 'u_ub.pt'))

u_nominal = list(u_nominal_jit.parameters())[0].cpu().numpy()
u_lb = list(u_lb_jit.parameters())[0].cpu().numpy()
u_ub = list(u_ub_jit.parameters())[0].cpu().numpy()

sample_keys = ['action_library', 'gaussian_unif', 'gaussian_walk']

samples_jit = {k: torch.jit.load(os.path.join(data_dir, '{}.pt'.format(k))) for k in sample_keys}
samples = {k: list(v.parameters())[0].cpu().numpy() for k, v in samples_jit.items()}

## viz ##
fig, axs = plt.subplots(2, 4, figsize=(24, 12))
for i in range(axs.shape[0]):
    axs[i, 0].set_ylabel('act dim {}'.format(i))
    for j in range(axs.shape[1]):
        axs[i, j].set_ylim(u_lb[0, i] - 0.1, u_ub[0, i] + 0.1)
        axs[i, j].axhline(u_lb[0, i], color='k')
        axs[i, j].axhline(u_ub[0, i], color='k')

colors = 'rgb'
for i in range(axs.shape[0]):
    for j, (k, x) in enumerate(samples.items()):
        axs[i, j].plot(x[0, :, :, i].T, c=colors[j], alpha=0.1)
        axs[i, -1].plot(x[0, :, :, i].T, c=colors[j], alpha=0.1)
        axs[i, j].set_title(k)

for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[i, j].plot(u_nominal[0, :, i], c='k', label='nominal')

plt.show()
