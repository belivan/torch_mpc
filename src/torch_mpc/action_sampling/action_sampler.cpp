#include <iostream>
#include <unordered_map>
#include <memory>
#include <string>
#include <stdexcept>
#include <chrono>
#include <variant>
#include <yaml-cpp/yaml.h>
#include <torch/torch.h>

#include "action_sampler.h"

int main()
{
    YAML::Node config = YAML::LoadFile("../../../configs/costmap_speedmap.yaml");
    const std::string device_config = config['common']['device'].as<std::string>();
    if (device_config == "cuda")
    {
        torch::Device device(torch::kCUDA, 0);
    }
    else if (device_config == "cpu")
    {
        torch::Device device(torch::kCPU);
    }
    else
    {
        throw std::runtime_error("Unknown device " + device_config);
    }

    const int B = config['common']['B'];
    const int H = config['common']['H'];
    const int M = config['common']['M'];
    const int dt = config['common']['dt'];
    
    std::unordered_map<std::string, std::unique_ptr<SamplingStrategy>> sampling_strategies;
    for(const auto& sv : config['sampling_strategies']['strategies'])
    {
        std::string type = sv['type'].as<std::string>();
        const std::vector<double> scale = sv['args']['scale'].as<std::vector<double>>();
        if (type == "UniformGaussian")
        {
            sampling_strategies[sv['label']] = std::make_unique<UniformGaussion>(scale, B, H, M, dt, device);
        }
        else if (type == "ActionLibrary")
        {
            sampling_strategies[sv['label']] = std::make_unique<ActionLibrary>(scale, B, H, M, dt, device);
        }
        else if (type == "GaussianWalk")
        {
            const std::unordered_map<std::string, std::variant<std::string, std::vector<double>>> initial_distribution;
            for(const auto& [k, v] : sv['args']['initial_distribution'])
            {
                if (k.as<std::string>() == "type")
                {
                    initial_distribution[k] = v.as<std::string>();
                }
                else if (k.as<std::string>() == "scale")
                {
                    initial_distribution[k] = v.as<std::vector<double>>();
                }
            }
            std::vector<double> alpha = sv['args']['alpha'].as<std::vector<double>>();
            sampling_strategies[sv['label']] = std::make_unique<GaussianWalk>(initial_distribution, scale, alpha, B, H, M, dt, device);
        }
        else
        {
            throw std::runtime_error("Unknown sampling strategy " + type);
        }
    }

    ActionSampler action_sampler(sampling_strategies);
    
    const torch::Tensor u_nominal = torch::stack({
        torch::linspace(0.0, 1.0, 50),
        torch::linspace(-0.5, -0.2, 50)
    }, 1).unsqueeze(0).to(torch::Device(device));

    const torch::Tensor u_lb = torch::tensor({0., -0.52}).view({1, 2}).to(torch::Device(device));
    const torch::Tensor u_ub = torch::tensor({1., 0.52}).view({1, 2}).to(torch::Device(device));

    auto t1 = std::chrono::high_resolution_clock::now();
    auto samples = action_sampler.sample_dict(u_nominal, u_lb, u_ub);
    auto t2 = std::chrono::high_resolution_clock::now();

    u_nominal = u_nominal.to(torch::kCPU);
    u_lb = u_lb.to(torch::kCPU);
    u_ub = u_ub.to(torch::kCPU);
    for(const auto& [k, v] : samples)
    {
        v.to(torch::kCPU);
    }

    std::cout << "took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/1000 << "s to sample\n";
}