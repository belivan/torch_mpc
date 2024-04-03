#ifndef ACTION_LIBRARY
#define ACTION_LIBRARY

#include "base.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

class ActionLibrary : public SamplingStrategy
{
private:
    torch::Tensor actlib;
public:
    ActionLibrary(std::string path, const int B, const int K, const int H, const int M, torch::Device device)
    : SamplingStrategy(B, K, H, M, device),
    actlib(setup_actlib(path)){}

    ~ActionLibrary(){}

    torch::Tensor setup_actlib(std::string path)
    {
        // torch::Tensor actlib_holder;
        // torch::load(actlib_holder, path, device);
        torch::jit::Module tensors = torch::jit::load(path);
        torch::Tensor actlib_holder = tensors.attr("cmds").toTensor();

        actlib_holder = actlib_holder.to(device);
        
        if (!actlib_holder.size(0) >= K)
        {
            std::stringstream msg;
            msg << "action library has dimension " << actlib_holder.size(0) << ". Expected " << K;
            throw std::runtime_error(msg.str());
        }

        if (!actlib_holder.size(1) >= H)
        {
            std::stringstream msg;
            msg << "action library has dimension " << actlib_holder.size(1) << ". Expected " << H;
            throw std::runtime_error(msg.str());
        }

        if (!actlib_holder.size(2) == M)
        {
            std::stringstream msg;
            msg << "action library has dimension " << actlib_holder.size(2) << ". Expected " << M;
            throw std::runtime_error(msg.str());
        }
        return actlib_holder.index({at::indexing::Slice(0, K),
                                   at::indexing::Slice(0, H),
                                   at::indexing::Slice(0, M)}).toType(at::kFloat);
    }

    torch::Tensor sample(const torch::Tensor &u_nominal, const torch::Tensor &u_lb, const torch::Tensor &u_ub)
    {
        auto samples = actlib.view({1,K,H,M});
        samples = samples.repeat({B, 1, 1, 1});

        return clip_samples(samples, u_lb, u_ub);
    }

    ActionLibrary& to(torch::Device &device) override
    {
        this->device = device;
        this->actlib = this->actlib.to(device);
        return *this;
    }
};

#endif

