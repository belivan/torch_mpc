import torch

from torch_mpc.action_sampling.sampling_strategies.base import SamplingStrategy
from torch_mpc.action_sampling.sampling_strategies.utils import clip_samples

class UniformGaussian(SamplingStrategy):
    """
    Sampling strategy that applies adds gaussian noise to the nominal sequence
    """
    def __init__(self, scale, B, K, H, M, device):
        """
        Args:
            scale: An m-element list of floats for scale on gaussian
        """
        super(UniformGaussian, self).__init__(B, K, H, M, device)

        self.scale = self.setup_scale(scale)

    def setup_scale(self, scale):
        """
        Set up/check scale on gaussian
        """
        scale = torch.Tensor(scale).to(self.device)

        #some checks
        assert len(scale) == self.M, "scale has dimension {}. Expected {}".format(actlib.shape[2], self.M)

        assert torch.all(scale >= 0), "got negative scale"

        return scale

    def sample(self, u_nominal, u_lb, u_ub):
        noise = torch.randn(self.B, self.K, self.H, self.M, device=self.device)
        noise = noise * self.scale.view(1, 1, 1, self.M)

        samples = u_nominal.view(self.B, 1, self.H, self.M) + noise
        samples_clip = clip_samples(samples, u_lb, u_ub)
        return samples_clip

    def to(self, device):
        self.device = device
        self.scale = self.scale.to(device)
        return self

if __name__ == '__main__':
    scale = [0.1, 0.5]

    B = 3
    K = 23
    H = 54
    M = 2

    sampling_strategy = UniformGaussian(scale=scale, B=B, K=K, H=H, M=M, device='cuda')

    nom = torch.zeros(B, H, M, device=sampling_strategy.device)
    ulb = -torch.ones(B, M, device=sampling_strategy.device)
    uub = torch.ones(B, M, device=sampling_strategy.device)

    ulb[0] *= 0.1
    uub[0] *= 0.1


    samples = sampling_strategy.sample(nom, ulb, uub)

    print('sample shape:\n', samples.shape)
    print('sample_lb:\n', samples.min(dim=1)[0].min(dim=1)[0])
    print('sample_ub:\n', samples.max(dim=1)[0].max(dim=1)[0])
