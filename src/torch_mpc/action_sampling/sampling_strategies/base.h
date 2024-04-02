#ifndef BASE_H_IS_INCLUDED
#define BASE_H_IS_INCLUDED

/*
Base class for sampling of actions for sampling MPC. All sampling strategies
must support the following:

Given a nominal sequence of shape B x T x M and action limits (B x M), 
return a Tensor of shape B x N x T x M

where:
    B: number of optimizations (i.e. instances of the MPC problem)
    N: number of samples to draw per optimization
    T: number of timesteps
    M: action dimension

Also, note that the number of samples to produce must be pre-sepcified so
that we can pre-allocate tensors
*/

#include <torch/torch.h>  // to be linked in CMakeLists.txt with libtorch

class SamplingStrategy:
{
friend class UniformGaussian;
friend class GaussianWalk;
private:
    /*
    Args:
        B: Number of optimizations to run
        K: Number of samples to generate per optimization
        H: Number of timesteps
        M: Action dimension
        device: device to compute samples on
    */
    // check what these variables are
    int B;
    int K;
    int H;
    int M;
    torch::Device device;
    double sigma;
    
public:
    SamplingStrategy(int B, int K, int H, int M, const torch::Device device)
    {
        this->B = B;
        this->K = K;
        this->H = H;
        this->M = M;
        this->device = device;
    }
    virtual ~SamplingStrategy() {}

    virtual torch::Tensor sample(const torch::Tensor &u_nominal, const torch::Tensor &u_lb, const torch::Tensor &u_ub) = 0;
    virtual SamplingStrategy& to(const torch::Device &device) = 0;  // check if I need to return anything
};

#endif