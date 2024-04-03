import torch
from torch import nn
import os
import argparse

class TensorContainer(nn.Module):
    def __init__(self, tensor_dict):
        super().__init__()
        for key, value in tensor_dict.items():
            setattr(self, key, value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_fp', type=str, required=True, help='path to save actlib')
    parser.add_argument('--throttle_lim', type=float, required=False, default=[0., 1.], nargs=2, help='min/max throttle to sample')
    parser.add_argument('--nthrottle', type=int, help='number of throttles to sample')
    parser.add_argument('--steer_lim', type=float, required=False, default=[-0.52, 0.52], nargs=2, help='min/max steer to sample')
    parser.add_argument('--nsteer', type=int, help='number of steers to sample')
    parser.add_argument('--H', type=int, required=True, help='num timesteps to run')
    args = parser.parse_args()

    throttles = torch.linspace(args.throttle_lim[0], args.throttle_lim[1], args.nthrottle)
    steers = torch.linspace(args.steer_lim[0], args.steer_lim[1], args.nsteer)

    cmd_1step = torch.stack(torch.meshgrid(throttles, steers, indexing='ij'), dim=-1).view(-1, 2)
    cmds = cmd_1step.unsqueeze(1).tile(1, args.H, 1)

    tensor_dict = {
        'cmds': cmds
    }

    tensors = TensorContainer(tensor_dict)
    tensors = torch.jit.script(tensors)

    tensors.save(args.save_fp)
