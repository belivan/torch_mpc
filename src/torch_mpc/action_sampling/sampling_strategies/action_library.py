import torch

from torch_mpc.action_sampling.sampling_strategies.base import SamplingStrategy
from torch_mpc.action_sampling.sampling_strategies.utils import clip_samples

class ActionLibrary(SamplingStrategy):
    """
    Sampling strategy that returns a precomputed set of actions
    """
    def __init__(self, path, B, K, H, M, device):
        """
        Args:
            path: path to load action 
        """
        super(ActionLibrary, self).__init__(B, K, H, M, device)
        print('load actlib from {}'.format(path))

        self.actlib = self.setup_actlib(path)

    def setup_actlib(self, path):
        """
        Reformat actlib to match sampling dimensions
        """
        actlib = torch.load(path, map_location='cpu')

        #some checks
        assert actlib.shape[0] >= self.K, "action library has {} samples. Need at least {}".format(actlib.shape[0], self.K)

        assert actlib.shape[1] >= self.H, "action library has {} steps. Need at least {}".format(actlib.shape[1], self.H)

        assert actlib.shape[2] == self.M, "action library has dimension {}. Expected {}".format(actlib.shape[2], self.M)

        return actlib[:self.K, :self.H, :self.M].to(self.device).float()

    def sample(self, u_nominal, u_lb, u_ub):
        samples = torch.tile(self.actlib.view(1, self.K, self.H, self.M), [self.B, 1, 1, 1])

        samples_clip = clip_samples(samples, u_lb, u_ub)

        return samples_clip

    def to(self, device):
        self.device = device
        self.actlib = self.actlib.to(device)
        return self

if __name__ == '__main__':
    actlib_path = '/home/atv/physics_atv_ws/src/control/torch_mpc/data/action_libraries/yamaha_atv/H75_throttle_steer.pt'

    B = 3
    K = 23
    H = 54
    M = 2

    sampling_strategy = ActionLibrary(path=actlib_path, B=B, K=K, H=H, M=M, device='cuda')

    nom = torch.zeros(B, H, M, device=sampling_strategy.device)
    ulb = -torch.ones(B, M, device=sampling_strategy.device)
    uub = torch.ones(B, M, device=sampling_strategy.device)

    ulb[0] *= 0.1
    uub[0] *= 0.1


    samples = sampling_strategy.sample(nom, ulb, uub)

    print('sample shape:\n', samples.shape)
    print('sample_lb:\n', samples.min(dim=1)[0].min(dim=1)[0])
    print('sample_ub:\n', samples.max(dim=1)[0].max(dim=1)[0])
