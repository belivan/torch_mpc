#include <iostream>
#include <vector>
#include <torch/torch.h>

#include "uniform_gaussian.h"

int main()
{
    const std::vector<double> scale = {0.1, 0.5};

    const int B = 3;
    const int K = 23;
    const int H = 54;
    const int M = 2;

    auto sampling_strategy = UniformGaussian(scale, B, K, H, M, torch::Device(torch::kCPU));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(sampling_strategy.device);
    const torch::Tensor nom = torch::zeros({B, H, M}, options);
    const torch::Tensor ulb = torch::ones({B, M}, options);
    const torch::Tensor uub = torch::ones({B, M}, options);

    ulb[0] *= 0.1;
    uub[0] *= 0.1;

    auto samples = sampling_strategy.sample(nom, ulb, uub);

    std::cout << samples.sizes() << std::endl;

    // auto sample_min = std::get<0>(samples.min(1)).min(1);
    // auto sample_max = std::get<0>(samples.max(1)).max(1);

    // std::cout << sample_min << std::endl;
    // std::cout << sample_max << std::endl;

    return 0;
}