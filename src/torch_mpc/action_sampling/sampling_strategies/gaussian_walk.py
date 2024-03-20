import torch

from torch_mpc.action_sampling.sampling_strategies.base import SamplingStrategy
from torch_mpc.action_sampling.sampling_strategies.utils import clip_samples

class GaussianWalk(SamplingStrategy):
    """
    Sampling strategy that applies adds gaussian noise to the nominal sequence
    """
    def __init__(self, initial_distribution, scale, alpha, B, K, H, M, device):
        """
        Args:
            initial_distribution: dict of dtype: params for initial dist from random walk
            scale: An m-element list of floats for scale on gaussian walk
            d_scale: An m-element list of floats for d_scale on gaussian walk
            alpha: An m-element list for low-passing the random walk
        """
        super(GaussianWalk, self).__init__(B, K, H, M, device)
        
        self.initial_distribution = self.setup_initial_distribution(initial_distribution)
        self.scale = self.setup_scale(scale)
        self.alpha = self.setup_alpha(alpha)

    def setup_initial_distribution(self, initial_distribution):
        valid_types = ['uniform', 'gaussian']
        assert initial_distribution['type'] in valid_types, "expected type to be in {}, got {}.".format(valid_types, initial_distribution['type'])

        if initial_distribution['type'] == 'gaussian':
            initial_distribution['scale'] = self.setup_scale(initial_distribution['scale'])

        return initial_distribution

    def setup_scale(self, scale):
        """
        Set up/check scale on gaussian
        """
        scale = torch.Tensor(scale).to(self.device)

        #some checks
        assert len(scale) == self.M, "scale has dimension {}. Expected {}".format(actlib.shape[2], self.M)

        assert torch.all(scale >= 0), "got negative scale"

        return scale

    def setup_alpha(self, alpha):
        """
        Set up/check scale on gaussian
        """
        alpha = torch.Tensor(alpha).to(self.device)

        #some checks
        assert len(alpha) == self.M, "alpha has dimension {}. Expected {}".format(actlib.shape[2], self.M)

        assert torch.all(alpha >= 0) and torch.all(alpha <= 1.), "got alpha not in [0, 1]"

        return alpha

    def sample(self, u_nominal, u_lb, u_ub):
        _scale = self.scale.view(1, 1, 1, self.M)
        _alpha = self.alpha.view(1, 1, 1, self.M)

        _ulb = u_lb.view(self.B, 1, 1, self.M) - u_nominal.view(self.B, 1, self.H, self.M)
        _uub = u_ub.view(self.B, 1, 1, self.M) - u_nominal.view(self.B, 1, self.H, self.M)

        noise_init = self.sample_initial_distribution(u_nominal, u_lb, u_ub)

        walk = [noise_init]
        dz = torch.randn(self.B, self.K, self.H-1, self.M, device=self.device) * _scale #[B x N x T x M]

        for t in range(self.H-1):
            walk_next = walk[-1] + (1-_alpha) * dz[:, :, [t], :]
            #enforce bounds at next timestep
            walk_next = walk_next.clip(_ulb[:, :, [t+1]], _uub[:, :, [t+1]])
            walk.append(walk_next)

        walk = torch.cat(walk, dim=2)
        samples = u_nominal.view(self.B, 1, self.H, self.M) + walk

        return samples

    def sample_initial_distribution(self, u_nominal, u_lb, u_ub):
        _ulb = u_lb.view(self.B, 1, 1, self.M) - u_nominal[:, 0].view(self.B, 1, 1, self.M)
        _uub = u_ub.view(self.B, 1, 1, self.M) - u_nominal[:, 0].view(self.B, 1, 1, self.M)
        if self.initial_distribution['type'] == 'uniform':
            noise = torch.rand(self.B, self.K, 1, self.M, device=u_nominal.device)
            noise = _ulb + (_uub-_ulb)*noise
            return noise

        elif self.initial_distribution['type'] == 'gaussian':
            noise = torch.randn(self.B, self.K, 1, self.M, device=self.device)
            noise = noise * self.initial_distribution['scale'].view(1, 1, 1, self.M)

            #keep samples in control lims
            noise_clip = noise.clip(_ulb, _uub)
            return noise_clip
        else:
            print('unsupported initial distribution {}'.format(self.initial_distribution['type']))
            exit(1)

    def to(self, device):
        self.device = device
        self.scale = self.scale.to(device)
        self.alpha = self.alpha.to(device)

        if self.initial_distribution['type'] == 'gaussian':
            self.initial_distribution['scale'] = self.initial_distribution['scale'].to(device)

        return self

if __name__ == '__main__':
    scale = [0.1, 0.5]
    alpha = [0.0, 0.9]
    initial_distribution = {
        'type': 'gaussian',
        'scale': [0.1, 0.1]
    }

    B = 3
    K = 23
    H = 54
    M = 2

    sampling_strategy = GaussianWalk(scale=scale, alpha=alpha, initial_distribution=initial_distribution, B=B, K=K, H=H, M=M, device='cuda')

    nom = torch.zeros(B, H, M, device=sampling_strategy.device)
    ulb = -torch.ones(B, M, device=sampling_strategy.device)
    uub = torch.ones(B, M, device=sampling_strategy.device)

    ulb[0] *= 0.1
    uub[0] *= 0.1


    samples = sampling_strategy.sample(nom, ulb, uub)

    print('sample shape:\n', samples.shape)
    print('sample_lb:\n', samples.min(dim=1)[0].min(dim=1)[0])
    print('sample_ub:\n', samples.max(dim=1)[0].max(dim=1)[0])
