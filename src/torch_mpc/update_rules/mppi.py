import torch

class MPPI:
    """
    Implement the MPPI update rule
    """
    def __init__(self, temperature):
        self.temperature = temperature

    def update(self, action_sequences, costs):
        ## Get minimum cost and obtain normalization constant
        beta = torch.min(costs, dim=-1, keepdims=True)[0]
        eta  = torch.sum(torch.exp(-1/self.temperature*(costs - beta)), keepdims=True, axis=-1)

        ## Get importance sampling weight
        sampling_weights = (1./eta) * ((-1./self.temperature) * (costs - beta)).exp()

        ## Get action sequence using weighted average
        controls = (action_sequences * sampling_weights.view(*sampling_weights.shape, 1, 1)).sum(dim=1)
        return controls, sampling_weights

