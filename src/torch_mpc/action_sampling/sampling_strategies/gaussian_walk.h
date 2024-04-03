#ifndef GAUSSIAN_WALK_H
#define GAUSSIAN_WALK_H

#include "base.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <variant>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

class GaussianWalk: public SamplingStrategy
{
    // Sampling strategy that applies adds gaussian noise to the nominal sequence
    private:
        // int B;
        // int K;
        // int H;
        // int M;
        // torch::Device device;
        torch::Tensor scale;
        std::unordered_map<std::string, std::variant<std::string, torch::Tensor>> initial_distribution;
        torch::Tensor alpha;

    public:
        GaussianWalk(std::unordered_map<std::string, std::variant<std::string, std::vector<double>>> initial_distribution,
                    const std::vector<double> scale,
                    const std::vector<double> alpha,
                    const int B, const int K,
                    const int H, const int M, torch::Device device)
        : SamplingStrategy(B, K, H, M, device),
        scale(setup_scale(scale)),
        initial_distribution(setup_initial_distribution(initial_distribution)),
        alpha(setup_alpha(alpha)){}

        torch::Tensor setup_scale(const std::vector<double> scale)
        {
            auto scale_holder = torch::tensor(scale, device);
            
            if (scale_holder.size(0) != M)
            {
                std::stringstream msg;
                msg << "scale has dimension " << scale_holder.size(0) << ". Expected " << M;
                throw std::runtime_error(msg.str());
            }

            if (!torch::all(scale_holder >= 0).item<bool>())
                throw std::runtime_error("got negative scale");

            return scale_holder;
        }

        std::unordered_map<std::string, std::variant<std::string, torch::Tensor>> setup_initial_distribution(std::unordered_map<std::string, std::variant<std::string, std::vector<double>>> initial_distribution)
        {
            std::vector<std::string> valid_types = {"uniform", "gaussian"};
            std::string type = std::get<std::string>(initial_distribution["type"]);
            if (std::find(valid_types.begin(), valid_types.end(), type) == valid_types.end())
            {
                std::string error_msg = "Invalid type for initial distribution. Expected one of [";
                for (size_t i = 0; i < valid_types.size(); ++i) {
                    error_msg += valid_types[i];
                    if (i < valid_types.size() - 1) {
                        error_msg += ", ";
                    }
    }           
                error_msg += "], got " + type;
                throw std::runtime_error(error_msg);    
            }

            std::unordered_map<std::string, std::variant<std::string, torch::Tensor>> initial_distribution_holder;
            initial_distribution_holder["type"] = type;
            if (std::get<std::string>(initial_distribution["type"]) == "uniform")
                initial_distribution_holder["scale"] = setup_scale(std::get<std::vector<double>>(initial_distribution["scale"]));
            
            return initial_distribution_holder;
        }

        torch::Tensor setup_alpha(const std::vector<double> alpha)
        {
            auto alpha_holder = torch::tensor(alpha, device);

            if (alpha_holder.numel() != M)
            {
                std::stringstream msg;
                msg << "alpha has dimension " << alpha_holder.numel() << ". Expected " << M;
                throw std::runtime_error(msg.str());
            }
            bool condition = torch::all(alpha_holder >= 0).item<bool>() && torch::all(alpha_holder <= 1).item<bool>();
            if (!condition)
                throw std::runtime_error("alpha must be in [0, 1]");

            return alpha_holder;
        }

        torch::Tensor sample(const torch::Tensor &u_nominal, const torch::Tensor &u_lb, const torch::Tensor &u_ub)
        {
            auto _scale = scale.view({1, 1, 1, M});
            auto _alpha = alpha.view({1, 1, 1, M});
 
            auto _ulb = u_lb.view({B, 1, 1, M}) - u_nominal.view({B, 1, H, M});
            auto _uub = u_ub.view({B, 1, 1, M}) - u_nominal.view({B, 1, H, M});

            auto noise_init = sample_initial_distribution(u_nominal, u_lb, u_ub);

            std::vector<torch::Tensor> walk = {noise_init}; 
            auto dz = torch::randn({B, K, H-1, M}, device) * scale; //[B x N x T x M]

            for (int t = 0; t < H-1; t++)
            {   
                auto walk_next = walk.back() + (1-alpha) * dz.narrow(2, t, 1); //dz[:, :, [t], :];
                // enforce bounds at the next timestep
                auto walk_next_holder = walk_next.clip(_ulb.narrow(2, t+1, 1), _uub.narrow(2, t+1, 1));  //_ulb[:, :, [t+1]], _uub[:, :, [t+1]]);
                walk.push_back(walk_next_holder);
            }

            // std::cout << walk.size() << std::endl;
            // std::cout << u_nominal.view({B, 1, H, M}).sizes() << std::endl;

            auto samples = u_nominal.view({B, 1, H, M}) + torch::cat(walk, 2);
            return samples;
        }

        torch::Tensor sample_initial_distribution(const torch::Tensor &u_nominal, const torch::Tensor &u_lb, const torch::Tensor &u_ub)
        {
            auto _ulb = u_lb.view({B, 1, 1, M}) - u_nominal.select(1,0).view({B, 1, 1, M});
            auto _uub = u_ub.view({B, 1, 1, M}) - u_nominal.select(1,0).view({B, 1, 1, M});


            if (std::get<std::string>(initial_distribution["type"]) == "uniform")
            {
                auto noise = torch::rand({B, K, 1, M}, device);
                noise = _ulb + noise * (_uub - _ulb);
                return noise;
            }
            else if (std::get<std::string>(initial_distribution["type"]) == "gaussian")
            {
                auto noise = torch::randn({B, K, 1, M}, device);
                noise = noise * std::get<torch::Tensor>(initial_distribution["scale"]).view({1, 1, 1, M});
                
                auto noise_clip = noise.clip(_ulb, _uub);
                return noise;
            }
            else
            {
                throw std::runtime_error("Invalid initial distribution type");
            }
        }

        GaussianWalk& to(torch::Device device) override
        {
            this->device = device;
            this->scale = this->scale.to(device);
            this->alpha = this->alpha.to(device);
            return *this;
        }
};

#endif