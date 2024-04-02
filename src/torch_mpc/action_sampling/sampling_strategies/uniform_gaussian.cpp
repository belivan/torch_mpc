#include <iostream>
#include <vector>
#include <torch/torch.h>

#include "uniform_gaussian.h"

int main()
{
    std::vector<double> scale = {0.1, 0.5};

    int B = 3;
    int K = 23;
    int H = 54;
    int M = 2;

    auto sampling_strategy = UniformGaussion(scale, B, K, H, M, torch::GPU(torch::kCUDA, 0));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(sampling_strategy.device);
    auto nom = torch::zeros({B, H, M}, options);
    auto ulb = torch::ones({B, M}, options);
    auto uub = torch::ones({B, M}, options);

    ulb[0] *= 0.1;
    uub[0] *= 0.1;

    auto samples = sampling_strategy.sample(nom, ulb, uub);

    std::cout << samples.sizes() << std::endl;

    auto sample_min = std::get<0>(samples.min(1)).min(1)[0];
    auto sample_max = std::get<0>(samples.max(1)).max(1)[0];

    std::cout << sample_min << std::endl;
    std::cout << sample_max << std::endl;

    return 0;
}
