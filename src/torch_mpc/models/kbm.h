// header file for kbm

#ifndef MODEL_KBMH_IS_INCLUDED
#define MODEL_KBMH_IS_INCLUDED

#include <torch/torch.h> // assuming we use libtorch for torch::Tensor
#include "base.h" // TODO: MAKE SURE I NEED THIS sadhfausdfhu
#include <iostream>
#include <vector>
#include <string>
#include <limits>

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

    torch::Device device;
public:
    std::vector<double> u_ub;
    std::vector<double> u_lb;

    KBM(double L = 3.0, double min_throttle = 0., 
        double max_throttle = 1., double max_steer = 0.3, 
        double dt = 0.1, const torch::Device& device = torch::kCPU)
        : L(L), dt(dt), device(device)
    {
        u_ub = { max_throttle, max_steer };
        u_lb = { min_throttle, -max_steer };

        std::cout << "KBM constructor" << std::endl;
        std::cout << u_lb[0] << std::endl;
        std::cout << u_lb[1] << std::endl;
        std::cout << u_ub[0] << std::endl;
        std::cout << u_ub[1] << std::endl;

        Model::u_lb = this->u_lb;
        Model::u_ub = this->u_ub;
    }

    // TODO : double check this implementation of dynamics, unsure how state / action are organized
    torch::Tensor dynamics(const torch::Tensor& state, const torch::Tensor& action) const
    {
        auto xyth = state.moveaxis(-1, 0);
        auto vd = action.moveaxis(-1, 0);

        // std::cout << "xyth" << xyth.sizes() << std::endl;
        // std::cout << "vd" << vd.sizes() << std::endl;

        auto x = xyth[0];
        auto y = xyth[1];
        auto th = xyth[2];

        auto v = vd[0];
        auto delta = vd[1];

        auto xd = v * torch::cos(th);
        auto yd = v * torch::sin(th);
        auto thd = v * torch::tan(delta) / L;
        // std::cout << "v" << v.sizes() << std::endl;
        // std::cout << "yd" << yd.sizes() << std::endl;
        auto result = torch::stack({xd, yd, thd}, -1);
        // std::cout << "result" << result.sizes() << std::endl;
        return result;
    }

    // probably should also be making const/non const versions
    // TODO: IS THIS EVEN REMOTELY CORRECT
    torch::Tensor predict(const torch::Tensor& state, const torch::Tensor& action) const override
    {
            // std::cout << "action: " << action.sizes() << std::endl;
            // std::cout << "state: " << state.sizes() << std::endl;
        // std::cout << state << std::endl;
        torch::Tensor k1 = dynamics(state, action);
        // std::cout << "k1" << k1.sizes() << std::endl;

        auto i1 = state + (dt / 2) * k1;
        torch::Tensor k2 = dynamics(i1, action);
        // std::cout << "k2" << k2.sizes() << std::endl;

        auto i2 = state + (dt / 2) * k2;
        torch::Tensor k3 = dynamics(i2, action);
        auto i3 = state + dt * k3;
        torch::Tensor k4 = dynamics(i3, action);
        auto result = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
        return result;
    }

    torch::Tensor rollout(const torch::Tensor& state, const torch::Tensor& actions) const
    {
        /*
        Expected shapes:
            state: [B1 x ... x Bn x xd]
            actions: [B1 x ... x Bn x T x ud]
            returns: [B1 x ... x Bn X T x xd]
        */
        std::vector<torch::Tensor> X;
        torch::Tensor curr_state = state.clone();

        int T = actions.size(-2);  // Get the size of the time dimension
        // std::cout << "T: " << T << std::endl;
        for (int t = 0; t < T; ++t)
        {
            // std::cout << "timestep: " << t << std::endl;
            torch::Tensor action = actions.index({ indexing::Ellipsis, t, indexing::Slice() });
            torch::Tensor next_state = predict(curr_state, action);
            X.push_back(next_state);
            curr_state = next_state.clone();
        }

        // Concatenate the tensors in X along the specified dimension
        return torch::stack(X, -2);
    }

    //TODO : is this returning a double? just want to double check
    torch::Tensor quat_to_yaw(const torch::Tensor& q)  // function not needed
    {
        auto q_permuted = q.permute({ q.dim() - 1, 0 });
        auto qx = q_permuted[0];
        auto qy = q_permuted[1];
        auto qz = q_permuted[2];
        auto qw = q_permuted[3];

        return torch::atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
    }

    int64_t observation_space() const {  // added
        // low = -np.ones(3).astype(float) * float('inf')
        // high = -low
        auto low = torch::ones({3}).to(torch::kFloat) * -std::numeric_limits<float>::infinity();
        auto high = -low;
        return low.size(0);
    }
    int64_t action_space() const {  // added
        return static_cast<int64_t>(u_lb.size());
    }
    /*
     def observation_space(self):
        low = -np.ones(4).astype(float) * float('inf')
        high = -low
        return gym.spaces.Box(low=low, high=high)
    */
    
    KBM& to(const torch::Device& device)
    {
        this->device = device;
        return *this;
    }
};

#endif