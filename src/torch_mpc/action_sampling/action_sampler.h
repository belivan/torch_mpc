#ifndef ACTION_SAMPLER_H
#define ACTION_SAMPLER_H

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

#include "sampling_strategies/base.h"
#include "sampling_strategies/uniform_gaussian.h"
#include "sampling_strategies/gaussian_walk.h"
#include "sampling_strategies/action_library.h"
#include <yaml-cpp/yaml.h>
#include "../algos/batch_sampling_mpc.h"

// add other classes

class ActionSampler
{
private:
    std::unordered_map<std::string, std::shared_ptr<SamplingStrategy>> sampling_strategies;

    // std::optional<int> B;
    // int K = 0;
    // std::optional<int> H;
    // std::optional<int> M;
    // std::optional<torch::Device> device;

    void make_action_sampler()
    {
        for (const auto& [k, strat] : sampling_strategies)
        {
            if (!B.has_value())
            {
                this->B = strat->B;
                std::cout << "B: " << B.value() << std::endl;
            }
            else if (this->B.value() != strat->B)
            {
                throw std::runtime_error("Mismatch in B. Got " + std::to_string(B.value()) + ", expected " + std::to_string(strat->B));
            }
            if (!H.has_value())
            {
                this->H = strat->H;
                std::cout << "H: " << H.value() << std::endl;
            }
            else if (this->H.value() != strat->H) 
            {
                throw std::runtime_error("Mismatch in H. Got " + std::to_string(H.value()) + ", expected " + std::to_string(strat->H));
            }
            if (!M.has_value())
            {
                this->M = strat->M;
                std::cout << "M: " << M.value() << std::endl;
            }
            else if (this->M.value() != strat->M) 
            {
                throw std::runtime_error("Mismatch in M. Got " + std::to_string(M.value()) + ", expected " + std::to_string(strat->M));
            }
            if (!this->device.has_value())
            {
                this->device = strat->device;
                std::cout << "Device: " << deviceToString(device.value()) << std::endl;
            }
            else if (deviceToString(this->device.value()) != deviceToString(strat->device))
            {
                std::string thisDevice = deviceToString(this->device.value());
                std::string stratDevice = deviceToString(strat->device);
                throw std::runtime_error("Mismatch in device. Got " + thisDevice + ", expected " + stratDevice);
            }

            this->K += strat->K;
        }
    }
    
public:
    // move to private, needed for testing script
    std::optional<int> B;
    int K = 0;
    std::optional<int> H;
    std::optional<int> M;
    std::optional<torch::Device> device;

    ActionSampler(std::unordered_map<std::string, std::shared_ptr<SamplingStrategy>>& sampling_strategies)
    : sampling_strategies(std::move(sampling_strategies))
    {
        make_action_sampler();
    }
    ~ActionSampler() {}

    std::unordered_map<std::string, torch::Tensor> sample_dict(const torch::Tensor &u_nominal, const torch::Tensor &u_lb, const torch::Tensor &u_ub) const
    {
        std::unordered_map<std::string, torch::Tensor> samples;
        for(const auto& [k, strat] : sampling_strategies)
        {
            samples[k] = strat->sample(u_nominal, u_lb, u_ub);
        }
        return samples;
    }

    torch::Tensor sample(const torch::Tensor &u_nominal, const torch::Tensor &u_lb, const torch::Tensor &u_ub) const
    {
        auto samples_map = this->sample_dict(u_nominal, u_lb, u_ub);
        std::vector<torch::Tensor> samples;
        for(const auto& [k, v] : samples_map)
        {
            samples.push_back(v);
        }
        return torch::cat(samples, 1);
    }
    ActionSampler& to(const torch::Device device)
    {
        this->device = device;
        for(const auto& [k, strat] : sampling_strategies)
        {
            strat->to(device);
        }
        return *this;
    }

    // Should be a util function
    std::string deviceToString(const torch::Device& device) const 
    {
        std::string deviceType;
        switch (device.type()) {
            case torch::kCPU: deviceType = "cpu"; break;
            case torch::kCUDA: deviceType = "cuda:" + std::to_string(device.index()); break;
            default: deviceType = "unknown";
        }
        return deviceType;
    }
};

#endif