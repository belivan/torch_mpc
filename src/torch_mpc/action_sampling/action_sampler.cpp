#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <memory> // if needed
#include <string>
#include <stdexcept> // For std::runtime_error
#include <optional> // For std::optional
#include <chrono>

#include "sampling_strategies/base.h"
#include "sampling_strategies/uniform_gaussian.cpp"
#include <yaml-cpp/yaml.h>

// add other classes

class ActionSampler
{
private:
    std::unordered_map<std::string, std::unique_ptr<SamplingStrategy>> sampling_strategies;

    std::optional<int> B;
    int K = 0;
    std::optional<int> H;
    std::optional<int> M;
    std::optional<torch::Device> device;
    
public:
    ActionSampler(std::unordered_map<std::string, std::unique_ptr<SamplingStrategy>>& sampling_strategies)
    {
        for (const auto& [k, strat] : sampling_strategies)
        {
            if (!this->B.has_value())
            {
                this->B = strat->B;
            }
            else if (B != strat->B)
            {
                throw std::runtime_error("Mismatch in B. Got " + std::to_string(B.value()) + ", expected " + std::to_string(strat->B));
            }
            if (!this->H.has_value())
            {
                this->H = strat->H;
            }
            else if (this->H != strat.H) 
            {
                throw std::runtime_error("Mismatch in H. Got " + std::to_string(H.value()) + ", expected " + std::to_string(strat->H));
            }
            if (!this->M.has_value())
            {
                this->M = strat->M;
            }
            else if (this->M != strat.M) 
            {
                throw std::runtime_error("Mismatch in M. Got " + std::to_string(M.value()) + ", expected " + std::to_string(strat->M));
            }
            if (!this->device.has_value())
            {
                this->device = strat->device;
            }
            else if (this->device != strat.device) 
            {
                throw std::runtime_error("Mismatch in device. Got " + this->device + ", expected " + strat->device);
            }

            this->K += strat->K;
        }
    }
    ~ActionSampler() {}

    std::unordered_map<std::string, torch::Tensor> sample_dict(const torch::Tensor &u_nominal, const torch::Tensor &u_lb, const torch::Tensor &u_ub) const
    {
        std::unordered_map<std::string, torch::Tensor>> samples;
        for(const auto& [k, strat] : sampling_strategies)
        {
            samples[k] = strat->sample(u_nominal, u_lb, u_ub);
        }
        return samples;
    }

    torch::Tensor sample(const torch::Tensor &u_nominal, const torch::Tensor &u_lb, const torch::Tensor &u_ub)
    {
        auto samples_map = this->sample_dict(u_nominal, u_lb, u_ub);
        std::vector<torch::Tensor> samples;
        for(const auto& [k, v] : samples_map)
        {
            samples.push_back(v);
        }
        return torch::cat(samples, 1);
    }
    ActionSampler& to(const torch::Device &device)
    {
        this->device = device;
        for(const auto& [k, strat] : sampling_strategies)
        {
            strat->to(device);
        }
        return *this;
    }
};

int main()
{
    YAML::Node config = YAML::LoadFile("../../../configs/costmap_speedmap.yaml");
    const std::string device = config['common']['device'].as<std::string>();
    const int B = config['common']['B'];
    const int H = config['common']['H'];
    const int M = config['common']['M'];
    const int dt = config['common']['dt'];
    
    std::unordered_map<std::string, std::unique_ptr<SamplingStrategy>> sampling_strategies;
    for(const auto& sv : config['sampling_strategies']['strategies'])
    {
        std::string type = sv['type'].as<std::string>();
        const std::vector<float> scale = sv['args']['scale'].as<std::vector<float>>();
        if (type == "UniformGaussian")
        {
            sampling_strategies['label'] = std::make_unique<UniformGaussion>(scale, B, H, M, dt, device);
        }
        else if (type == "ActionLibrary")
        {
            sampling_strategies['label'] = std::make_unique<NActionLibrary>(type);
        }
        else if (type == "gaussian_mixture")
        {
            sampling_strategies[type] = std::make_unique<GaussianMixtureSamplingStrategy>(type);
        }
        else
        {
            throw std::runtime_error("Unknown sampling strategy " + type);
        }
    }
    ActionSampler action_sampler();
    
    auto u_nominal = torch::stack({
        torch::linspace(0.0, 1.0, 50),
        torch::linspace(-0.5, -0.2, 50)
    }, 1).unsqueeze(0).to(torch::Device(device));

    auto u_lb = torch::tensor({0., -0.52}).view({1, 2}).to(torch::Device(device));
    auto u_ub = torch::tensor({1., 0.52}).view({1, 2}).to(torch::Device(device));

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