#include <iostream>
#include <vector>
#include <unordered_map>
#include <variant>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <torch/torch.h>

#include "gaussian_walk.h"

int main()
{
    std::vector<double> scale = {0.1, 0.5};
    std::vector<double> alpha = {0.1, 0.9};

    int B = 3;
    int K = 23;
    int H = 54;
    int M = 2;

    std::unordered_map<std::string, std::variant<std::string, std::vector<double>>> initial_distribution;
    initial_distribution['type'] = "uniform";
    initial_distribution['scale'] = {0.1, 0.5};

    auto sampling_strategy = GaussianWalk(initial_distribution, scale, B, K, alpha, H, M, torch::GPU(torch::kCUDA, 0));

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