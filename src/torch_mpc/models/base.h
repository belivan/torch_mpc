#ifndef MODEL_H_IS_INCLUDED
#define MODEL_H_IS_INCLUDED

#include <torch/torch.h> // fajsidofajosdfji error ashdfhadsf TODO: fix this

class Model
{
    /*
    Base model class
    */


public:
    
    virtual torch::Tensor predict() = 0; // TODO: make sure action/observation space not necessary
    virtual torch::Tensor rollout() = 0;
    virtual torch::Tensor get_observations() = 0;
    virtual torch::Tensor get_actions() = 0;
};

#endif