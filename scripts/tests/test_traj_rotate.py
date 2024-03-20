import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_mpc.cost_functions.cost_terms.utils import move_to_local_frame

if __name__ == '__main__':
    xs = torch.linspace(0., 3., 50)
    ys = torch.zeros_like(xs)
    ths = torch.linspace(-np.pi/2., np.pi/2, 20)

    trajs = torch.stack([
        xs.view(1, 50).tile(20, 1),
        ys.view(1, 50).tile(20, 1),
        ths.view(20, 1).tile(1, 50)
    ], dim=-1)

    trajs[..., 0] += torch.linspace(-10., 10., 20).view(20, 1)
    trajs[..., 1] += torch.linspace(10., -10., 20).view(20, 1)

    local_trajs = move_to_local_frame(trajs.unsqueeze(0)).squeeze(0)

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title('global frame')
    axs[1].set_title('local frame')

    for traj in trajs:
        axs[0].plot(traj[:, 0], traj[:, 1])
        axs[0].arrow(traj[0, 0], traj[0, 1], traj[0, 2].cos(), traj[0, 2].sin())

    for traj in local_trajs:
        axs[1].plot(traj[:, 0], traj[:, 1])
        axs[1].arrow(traj[0, 0], traj[0, 1], traj[0, 2].cos(), traj[0, 2].sin())

    for ax in axs:
        ax.set_aspect(1.)
    plt.show()
