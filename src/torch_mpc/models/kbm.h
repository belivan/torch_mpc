// header file for kbm

#ifndef MODEL_KBMH_IS_INCLUDED
#define MODEL_KBMH_IS_INCLUDED

#include <torch/torch.h> // assuming we use libtorch for torch::Tensor
#include "base.h" // TODO: MAKE SURE I NEED THIS sadhfausdfhu
#include <iostream>

namespace indexing = torch::indexing;

// will make this class KBM : public Model once class Model is done aaa
class KBM : public Model
{
    /*
    Kinematic bicycle model
    x = [x, y, th]
    u = [v, delta]
    xdot = [
        v * cos(th)
        v * sin(th)
        L / tan(delta)
    */
private:
    double L;
    double dt;

    std::string device;

    std::vector<double> u_ub;
    std::vector<double>u_lb; // unsure if this makes sense

public:
    // def __init__(self, L=3.0, min_throttle=0., max_throttle=1., max_steer=0.3, dt=0.1, device='cpu'):
    //     self.L = L
    //     self.dt = dt
    //     self.device = device
    //     self.u_ub = np.array([max_throttle, max_steer])
    //     self.u_lb = np.array([min_throttle, -max_steer])
    // KBMModel(double l = 3.0, double min_throttle = 0.0, double max_throttle = 1.0, double max_steer = 0.3, double d_t = 0.1, torch:Device dev = 'cpu')
    // {
    //     L = l;
    //     dt = d_t;
    //     u_ub = {max_throttle, max_steer};
    //     u_lb = {min_throttle, -max_steer};
    //     device = dev;
    // }

    KBM(double L = 3.0, double min_throttle = 0., double max_throttle = 1., double max_steer = 0.3, double dt = 0.1, std::string device = "cpu")
        : L(L), dt(dt), device(device)
    {
        u_ub = { max_throttle, max_steer };
        u_lb = { min_throttle, -max_steer };
    }

    // TODO : double check this implementation of dynamics, unsure how state / action are organized
    torch::Tensor dynamics(const torch::Tensor& state, const torch::Tensor& action) const
    {
        auto x = state.index({ indexing::Ellipsis, 0 });
        auto y = state.index({ indexing::Ellipsis, 1 });
        auto th = state.index({ indexing::Ellipsis, 2 });
        auto v = action.index({ indexing::Ellipsis, 0 });
        auto d = action.index({ indexing::Ellipsis, 1 });
        auto xd = v * th.cos();
        auto yd = v * th.sin();
        auto thd = v * torch::tan(d) / L;
        std::cout << "finished doing dynamics, just have to return it!" << std::endl;
        return torch::stack({ xd, yd, thd }, -1);
    }

    // probably should also be making const/non const versions
    // TODO: IS THIS EVEN REMOTELY CORRECT
    torch::Tensor predict(const torch::Tensor& state, const torch::Tensor& action) const override
    {
        std::cout << state << std::endl;
        torch::Tensor k1 = dynamics(state, action);
        std::cout << "first dynamics goes well" << std::endl;
        std::cout << "this is k1" << k1 << std::endl;

        std::cout << "state" << std::endl;
        std::cout << state << std::endl;
        auto inputhelper = state + dt / 2 * k1;
        std::cout << "this is state + dt/2 * k1" << std::endl;
        std::cout << inputhelper << std::endl;
        std::cout << " " << std::endl;
        auto input2 = state + (dt / 2) * k1;


        torch::Tensor k2 = dynamics(input2, action);
        std::cout << "second dynamics goes well" << std::endl;
        torch::Tensor k3 = dynamics(state + (dt / 2) * k2, action);
        std::cout << "third dynamics goes well" << std::endl;
        torch::Tensor k4 = dynamics(state + (dt * k3), action);
        std::cout << "fourth dynamics goes well" << std::endl;

        return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    }

    //TODO: double check the dimensions of X?
    //torch::Tensor rollout(const torch::Tensor& state, const torch::Tensor& actions) const override
    //{
    //    torch::Tensor X = torch::empty({ state.size(0), actions.size(1) }); // same # rows state, # cols action?
    //    auto curr_state = state.clone();
    //    for (int i = 0; i < actions.size(1); ++i) // not sure if this is the right dimension to loop by
    //    {
    //        torch::Tensor action = actions.index({ indexing::Ellipsis, i, indexing::Slice() });
    //        torch::Tensor next_state = predict(curr_state, action);
    //        std::cout << " rollout does a prediction of next state" << std::endl;
    //        X.select(1, i).copy_(next_state);
    //        curr_state = next_state.clone();
    //    }
    //    std::cout << " rollout is going to return" << std::endl;
    //    std::cout << X << std::endl;
    //    return X;
    //}
 
    //torch::Tensor rollout(const torch::Tensor& state, const torch::Tensor& actions) const override
    //{
    //    std::cout << "starting rollout" << std::endl;
    //    std::cout << "state size" << state.size(0) << std::endl;
    //    torch::Tensor X = torch::empty({ actions.size(0), state.size(0) }); // size: #samples x #time_steps x state_size
    //    std::cout << "made empty X tensor" << std::endl;
    //    auto curr_state = state.clone();
    //    std::cout << "clone curstate" << std::endl;
    //    for (int i = 0; i < actions.size(0); ++i)
    //    {
    //        torch::Tensor action = actions.index({ i, indexing::Slice() });
    //        torch::Tensor next_state = predict(curr_state, action);
    //        X.index_put_({ torch::indexing::Slice(), i }, next_state);
    //        curr_state = next_state.clone();
    //    }
    //    return X;
    //}

    torch::Tensor rollout(const torch::Tensor& state, const torch::Tensor& actions) const
    {
        std::vector<torch::Tensor> X;
        torch::Tensor curr_state = state.clone();
        int T = actions.size(-2);  // Get the size of the time dimension

        for (int t = 0; t < T; ++t)
        {
            torch::Tensor action = actions.index({ torch::indexing::Ellipsis, t, torch::indexing::Slice() });
            torch::Tensor next_state = predict(curr_state, action);
            X.push_back(next_state);
            curr_state = next_state.clone();
        }

        // Concatenate the tensors in X along the specified dimension
        return torch::stack(X, -2);
    }

    //TODO : is this returning a double? just want to double check
    torch::Tensor quat_to_yaw(const torch::Tensor& q)
    {
        auto q_permuted = q.permute({ q.dim() - 1, 0 });
        auto qx = q_permuted[0];
        auto qy = q_permuted[1];
        auto qz = q_permuted[2];
        auto qw = q_permuted[3];

        return torch::atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
    }

    /*
     def observation_space(self):
        low = -np.ones(4).astype(float) * float('inf')
        high = -low
        return gym.spaces.Box(low=low, high=high)
    */
};

#endif