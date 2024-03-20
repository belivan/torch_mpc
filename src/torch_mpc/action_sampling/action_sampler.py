import torch

from torch_mpc.action_sampling.sampling_strategies.action_library import ActionLibrary
from torch_mpc.action_sampling.sampling_strategies.uniform_gaussian import UniformGaussian
from torch_mpc.action_sampling.sampling_strategies.gaussian_walk import GaussianWalk

class ActionSampler:
    """
    Main driver class for sampling actions
    """
    def __init__(self, sampling_strategies):
        self.sampling_strategies = sampling_strategies

        self.B = None
        self.K = 0
        self.H = None
        self.M = None
        self.device = None

        for k,strat in self.sampling_strategies.items():
            if self.B is None:
                self.B = strat.B
            else:
                assert self.B == strat.B, "Mismatch in B. Got {}, expected {}".format(self.B, strat.B)

            if self.H is None:
                self.H = strat.H
            else:
                assert self.H == strat.H, "Mismatch in H. Got {}, expected {}".format(self.H, strat.H)

            if self.M is None:
                self.M = strat.M
            else:
                assert self.M == strat.M, "Mismatch in M. Got {}, expected {}".format(self.M, strat.M)

            if self.device is None:
                self.device = strat.device
            else:
                assert self.device == strat.device, "Mismatch in device. Got {}, expected {}".format(self.device, strat.device)

            self.K += strat.K

    def sample_dict(self, u_nominal, u_lb, u_ub):
        samples = {k:v.sample(u_nominal, u_lb, u_ub) for k,v in self.sampling_strategies.items()}
        return samples

    def sample(self, u_nominal, u_lb, u_ub):
        samples = self.sample_dict(u_nominal, u_lb, u_ub)
        return torch.cat([v for v in samples.values()], dim=1)

    def to(self, device):
        self.device = device
        self.sampling_strategies = {k:v.to(device) for k,v in self.sampling_strategies.items()}
        return self

if __name__ == '__main__':
    import yaml
    import time
    import matplotlib.pyplot as plt

    config_fp = '/home/atv/physics_atv_ws/src/control/torch_mpc/configs/test_config.yaml'
    config = yaml.safe_load(open(config_fp, 'r'))
    device = config['common']['device']

    action_sampler = ActionSampler(config)

    u_nominal = torch.stack([
        torch.linspace(0., 1., 50),
        torch.linspace(-0.5, -0.2, 50)
    ], dim=-1).unsqueeze(0).to(device)

    u_lb = torch.tensor([0., -0.52], device=device).view(1, 2)
    u_ub = torch.tensor([1., 0.52], device=device).view(1, 2)

    t1 = time.time()
    samples = action_sampler.sample_dict(u_nominal, u_lb, u_ub)
    t2 = time.time()

    u_nominal = u_nominal.to('cpu')
    u_lb = u_lb.to('cpu')
    u_ub = u_ub.to('cpu')
    samples = {k:v.to('cpu') for k,v in samples.items()}

    print('took {:.4f}s to sample'.format(t2-t1))

    ## viz ##
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    for i in range(axs.shape[0]):
        axs[i, 0].set_ylabel('act dim {}'.format(i))
        for j in range(axs.shape[1]):
            axs[i, j].set_ylim(u_lb[0, i]-0.1, u_ub[0, i]+0.1)
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

    fig.suptitle('sampling (took {:.4f}s)'.format(t2-t1))
    plt.show() 
