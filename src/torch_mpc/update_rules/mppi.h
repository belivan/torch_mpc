#ifndef MPPPI_IS_INCLUDED
#define MPPPI_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <string>
#include <stdexcept>
#include <optional>
#include <chrono>
#include <variant>

class MPPI {
    /*
    Implement the MPPI update rule
    */

private:
    float temperature;
public:
    MPPI(float temperature) : temperature(temperature) {}

    torch::Tensor update(torch::Tensor action_sequences, torch::Tensor costs) {
        // Get minimum cost and obtain normalization constant
        auto beta = torch::min(costs, /*dim=*/-1, /*keepdim=*/true).values();
        auto eta = torch::sum(torch::exp(-1 / temperature * (costs - beta)), /*axis=*/-1, /*keepdim=*/true);

        // Get importance sampling weight
        auto sampling_weights = (1.0 / eta) * ((-1.0 / temperature) * (costs - beta)).exp();

        // Get action sequence using weighted average
        auto controls = (action_sequences * sampling_weights.view({ sampling_weights.sizes(), 1, 1 })).sum(/*dim=*/1);
        return controls;
    }


};

#endif // MPPPI_IS_INCLUDED