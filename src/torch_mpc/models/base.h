#ifndef MODEL_H_IS_INCLUDED
#define MODEL_H_IS_INCLUDED

#include <torch/torch.h> // fajsidofajosdfji error ashdfhadsf TODO: fix this
#include <vector>
#include <iostream>

class Model
{
    /*
    Base model class
    */


public:
    std::vector<double> u_ub;
    std::vector<double> u_lb;

    Model() = default;
    // Model(std::vector<double> u_ub, std::vector<double> u_lb) 
    // {
    //     this->u_ub = u_ub;
    //     this->u_lb = u_lb;
    // }

    virtual torch::Tensor predict(const torch::Tensor& state, const torch::Tensor& action) const = 0;
    virtual torch::Tensor rollout(const torch::Tensor& state, const torch::Tensor& actions) const = 0;
    virtual int64_t observation_space() const = 0;
    virtual int64_t action_space() const = 0;
    virtual Model& to(const torch::Device& device) = 0;
}; 

#endif