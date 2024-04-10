#ifndef MODEL_H_IS_INCLUDED
#define MODEL_H_IS_INCLUDED

#include <torch/torch.h> // fajsidofajosdfji error ashdfhadsf TODO: fix this

class Model
{
    /*
    Base model class
    */


public:
    
    //virtual torch::Tensor predict() = 0; // TODO: make sure action/observation space not necessary
    //virtual torch::Tensor rollout() = 0;
    virtual torch::Tensor predict(const torch::Tensor& state, const torch::Tensor& action) const = 0;
    virtual torch::Tensor rollout(const torch::Tensor& state, const torch::Tensor& actions) const = 0;
};

#endif