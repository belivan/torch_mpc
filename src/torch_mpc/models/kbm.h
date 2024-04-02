// header file for kbm
#include <torch/torch.h> // assuming we use libtorch for torch::Tensor
#include "base.h" // TODO: MAKE SURE I NEED THIS sadhfausdfhu

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

    KBM(float L=3.0, float min_throttle=0., float max_throttle=1., float max_steer=0.3, float dt=0.1, std::string device="cpu")
    : L(L), dt(dt), device(device) 
    {
        u_ub = {max_throttle, max_steer};
        u_lb = {min_throttle, -max_steer};
    }

    // TODO : double check this implementation of dynamics, unsure how state / action are organized
    torch::Tensor dynamics(const torch::Tensor &state, const torch::Tensor &action) const
    {
        // TODO : what are you doing
        // torch::Tensor state_permute = state.permute({state.dim() - 1, 0}).contiguous(); 

        /* Added by Anton
        // Assuming state is a tensor with at least 3 dimensions
        auto state_permuted = state.permute({2, 0, 1}); // This rearranges the axes of the tensor

        // Assuming the first dimension after permutation corresponds to what you want in x, y, th
        auto x = state_permuted.select(0, 0); // Equivalent to state_permuted[0] in Python
        auto y = state_permuted.select(0, 1); // Equivalent to state_permuted[1] in Python
        auto th = state_permuted.select(0, 2); // Equivalent to state_permuted[2] in Python
        */

        // going to try this for now fhausdifhsadf
        // auto state_permuted = state.permute({2, 0, 1});
        // auto x = state_permuted.select(0, 0);
        // auto y = state_permuted.select(0, 1); 
        // auto th = state_permuted.select(0, 2);

        // auto action_permuted = action.permute({2, 0, 1});
        // auto v = action_permuted.select(0, 0);
        // auto d = action_permuted.select(0, 1); 

        // double x, y, th = state.moveaxis(-1, 0);
       // only based on v, d, th
    //    double x = state_permute[0].item<double>(); // do these need to be torch tensors even, i would like them to be doubles or something
        // torch
        // could maybe consider implementation like double x = state[0].item<double>(); ??? if that works
        // double v, d = action.moveaxis(-1, 0);
        // double d;
        // double th = state_permute[2].item<double>(); // please tell me how state is organized
        // return torch::Tensor({v * std::cos(th), v * std::sin(th), v * std::tan(d) / L});

        auto x = state.index({indexing::Ellipsis, 0});
        auto y = state.index({indexing::Ellipsis, 1});
        auto th = state.index({indexing::Ellipsis, 2});
        auto v = action.index({indexing::Ellipsis, 0});
        auto d = action.index({indexing::Ellipsis, 1});
        auto xd = v * th.cos();
        auto yd = v * th.sin();
        auto thd = v * torch::tan(d) / L;
        return torch::stack({xd, yd, thd}, -1);
    }

    // probably should also be making const/non const versions
    // TODO: IS THIS EVEN REMOTELY CORRECT
    torch::Tensor predict(const torch::Tensor &state, const torch::Tensor &action) const override
    {
        torch::Tensor k1 = dynamics(state, action);
        torch::Tensor k2 = dynamics(state + (dt / 2) * k1, action);
        torch::Tensor k3 = dynamics(state + (dt / 2) * k2, action);
        torch::Tensor k4 = dynamics(state + (dt * k3), action);

        return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    }

    // TODO: double check the dimensions of X?
    // torch::Tensor rollout(const torch::Tensor &state, const torch::Tensor &actions) const override
    // {
    //     torch::Tensor X = torch::empty({state.size(0), actions.size(1)}) // same # rows state, # cols action?
    //     torch::Tensor curr_state = state;
    //     for (int i = 0; i < actions.size(1), ++i)
    //     {
    //         torch::Tensor action = actions.select(1, i);
    //         torch::Tensor next_state = predict(curr_state, action);
    //         X.select(1, i) = next_state;
    //         curr_state = next_state.clone();
    //     }
    //     return X;
    // }

    // think this implementation of rollout may be faster, if not then try using other one ajsidofjisd
    torch::Tensor rollout(torch::Tensor state, torch::Tensor actions) 
    {
        std::vector<torch::Tensor> X;
        torch::Tensor curr_state = state;
        for (int t = 0; t < actions.size(-2); ++t) {
            auto action = actions.index({indexing::Ellipsis, t, indexing::Ellipsis});
            auto next_state = predict(curr_state, action);
            X.push_back(next_state);
            curr_state = next_state.clone();
        }
    return torch::stack(X, -2);
    }

    //TODO : is this returning a double? just want to double check
    double quat_to_yaw(const torch::Tensor &q) const
    {
        auto q_permuted = q.permute({q.dim() - 1, 0});
        auto qx = q_permuted[0];
        auto qy = q_permuted[1];
        auto qz = q_permuted[2];
        auto qw = q_permuted[3];
        // return torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));

        // auto qx = q.index({indexing::Ellipsis, 0});
        // auto qy = q.index({indexing::Ellipsis, 1});
        // auto qz = q.index({indexing::Ellipsis, 2});
        // auto qw = q.index({indexing::Ellipsis, 3});
        return torch::atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz));
    }


    // TODO: HOW TO DO SLICE INDEXING IN C++
    // HELP JASDOIJFJIOASDFIJOFJIOOIJ
    torch::Tensor get_observations(const torch::Tensor &batch) const override
    {
        // torch::Tensor state = batch['state']; // can i index like this in c++?

        // // 

        // torch::Tensor x = state.index({torch::Slice(), torch::Slice(), torch::Slice(), 0}); // how do i do ... indexing in c++
        // torch::Tensor y = state.index({torch::Slice(), torch::Slice(), torch::Slice(), 1});
        // torch::Tensor q = state.slice(0, 3, 7); // ok this slicing makes no sense

        // this should be the proper way to index, assuming that namespace was done correctly
        auto state = batch.index({"state"});
        if (state.dim() == 1) {
            return get_observations(torch::indexing::dict_map(batch, [](torch::Tensor x){ return x.unsqueeze(0); })).squeeze();
        }

        auto x = state.index({indexing::Ellipsis, 0});
        auto y = state.index({indexing::Ellipsis, 1});
        auto q = state.index({indexing::Ellipsis, Slice(3, 7)});
        auto yaw = quat_to_yaw(q);
        return torch::stack({x, y, yaw}, -1);
    }

    // is this function necessary to have
    torch::Tensor get_actions(const torch::Tensor &batch) const override
    {
        return batch;
    }

    

};