#ifndef UNIFORM_GAUSSIAN_H
#define UNIFORM_GAUSSIAN_H

#include "base.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <unordered_map>

class UniformGaussian: public SamplingStrategy
{
    /*
    Sampling strategy that applies adds gaussian noise to the nominal sequence
    */
    private:
        // int B;
        // int K;
        // int H;
        // int M;
        // torch::Device device;
        torch::Tensor scale;
    public:
        UniformGaussian(const std::vector<double> scale, const int B, const int K, const int H, const int M, torch::Device device)
        : SamplingStrategy(B, K, H, M, device),
        scale(setup_scale(scale)){}  // check if this is the right way to do this

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

        torch::Tensor sample(const torch::Tensor &u_nominal, const torch::Tensor &u_lb, const torch::Tensor &u_ub)
        {
            auto noise = torch::randn({B, K, H, M}, device);
            noise *= scale.view({1,1,1,M});

            auto samples = u_nominal.view({B, 1, H, M}) + noise;
            auto samples_clip = samples.clip(u_lb, u_ub);

            return samples_clip;
        }

        UniformGaussian& to(torch::Device &device) override
        {
            this->device = device;
            this->scale = this->scale.to(device);
            return *this;
        }
};

#endif