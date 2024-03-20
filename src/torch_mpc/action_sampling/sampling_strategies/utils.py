import torch

def clip_samples(samples, ulb, uub):
    """
    Clip samples to fall in the range given by ulb, uub
    Args:
        samples: [B x N x H x M] Tensor of samples to clip
        ulb: [B x M] lower limit
        uub: [B x M] upper limit
    """
    B = samples.shape[0]
    M = samples.shape[-1]
    _ulb = ulb.view(B, 1, 1, M)
    _uub = uub.view(B, 1, 1, M)
    samples_clip = samples.clip(_ulb, _uub)

    return samples_clip
