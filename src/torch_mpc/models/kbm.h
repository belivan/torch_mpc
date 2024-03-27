// header file for kbm
#include <base.h> // TODO: MAKE SURE I NEED THIS sadhfausdfhu
#include <torch/torch.h> // assuming we use libtorch for torch::Tensor

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

    std::vector<double> u_ub;
    std::vector<double>u_lb; // unsure if this makes sense

public:
    // def __init__(self, L=3.0, min_throttle=0., max_throttle=1., max_steer=0.3, dt=0.1, device='cpu'):
    //     self.L = L
    //     self.dt = dt
    //     self.device = device
    //     self.u_ub = np.array([max_throttle, max_steer])
    //     self.u_lb = np.array([min_throttle, -max_steer])
    KBMModel(double l = 3.0, double min_throttle = 0.0, double max_throttle = 1.0, double max_steer = 0.3, double d_t = 0.1, torch:Device dev = 'cpu')
    {
        L = l;
        dt = d_t;
        u_ub = {max_throttle, max_steer};
        u_lb = {min_throttle, -max_steer};
        device = dev;
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
        auto state_permuted = state.permute({2, 0, 1});
        auto x = state_permuted.select(0, 0);
        auto y = state_permuted.select(0, 1); 
        auto th = state_permuted.select(0, 2);

        auto action_permuted = action.permute({2, 0, 1});
        auto v = action_permuted.select(0, 0);
        auto d = action_permuted.select(0, 1); 

        // double x, y, th = state.moveaxis(-1, 0);
       // only based on v, d, th
    //    double x = state_permute[0].item<double>(); // do these need to be torch tensors even, i would like them to be doubles or something
        // torch
        // could maybe consider implementation like double x = state[0].item<double>(); ??? if that works
        // double v, d = action.moveaxis(-1, 0);
        // double d;
        // double th = state_permute[2].item<double>(); // please tell me how state is organized
        return torch::Tensor({v * std::cos(th), v * std::sin(th), v * std::tan(d) / L});
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
    torch::Tensor rollout(const torch::Tensor &state, const torch::Tensor &actions) const override
    {
        torch::Tensor X = torch::empty({state.size(0), actions.size(1)}) // same # rows state, # cols action?
        torch::Tensor curr_state = state;
        for (int i = 0; i < actions.size(1), ++i)
        {
            torch::Tensor action = actions.select(1, i);
            torch::Tensor next_state = predict(curr_state, action);
            X.select(1, i) = next_state;
            curr_state = next_state.clone();
        }
        return X;
    }

    //TODO : is this returning a double? just want to double check
    double quat_to_yaw(const torch::Tensor &q) const
    {
        double qx = 
        double qy = 
        double qz = 
        double qw = 
        return torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
    }

    // TODO: HOW TO DO SLICE INDEXING IN C++
    // HELP JASDOIJFJIOASDFIJOFJIOOIJ
    torch::Tensor get_observations(const torch::Tensor &batch) const override
    {
        torch::Tensor state = batch['state']; // can i index like this in c++?

        // 

        torch::Tensor x = state.index({torch::Slice(), torch::Slice(), torch::Slice(), 0}); // how do i do ... indexing in c++
        torch::Tensor y = state.index({torch::Slice(), torch::Slice(), torch::Slice(), 1});
        torch::Tensor q = state.slice(0, 3, 7); // ok this slicing makes no sense
    }

    // is this function necessary to have
    torch::Tensor get_actions(const torch::Tensor &batch) const override
    {
        return batch;
    }

    

};