#ifndef MPPPI_IS_INCLUDED
#define MPPPI_IS_INCLUDED

#include <torch/torch.h>

class MPPI {
    /*
    Implement the MPPI update rule
    */

private:
    double temperature;
public:
    MPPI(const double temperature) : temperature(temperature) {}

    std::pair<torch::Tensor, torch::Tensor> update(const torch::Tensor& action_sequences, const torch::Tensor& costs) const {
        // Get minimum cost and obtain normalization constant
        auto beta = std::get<0>(torch::min(costs, /*dim=*/-1, /*keepdim=*/true));
        auto eta = torch::sum(torch::exp(-1 / temperature * (costs - beta)), /*axis=*/-1, /*keepdim=*/true);

        // Get importance sampling weight
        auto sampling_weights = (1.0 / eta) * torch::exp((-1.0 / temperature) * (costs - beta));

        // Get action sequence using weighted average
        auto controls = (action_sequences * sampling_weights.view({-1, 1, 1 })).sum(/*dim=*/1);
        return {controls, sampling_weights};
    }
};

#endif // MPPPI_IS_INCLUDED