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

    KBM(double L=3.0, double min_throttle=0., double max_throttle=1., double max_steer=0.3, double dt=0.1, std::string device="cpu")
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
        std::cout << "finished doing dynamics, just have to return it!" << std::endl;
        return torch::stack({xd, yd, thd}, -1);
    }

    // probably should also be making const/non const versions
    // TODO: IS THIS EVEN REMOTELY CORRECT
    torch::Tensor predict(const torch::Tensor &state, const torch::Tensor &action) const override
    {
        std::cout << state << std::endl;
        torch::Tensor k1 = dynamics(state, action);
        std::cout << "first dynamics goes well" << std::endl;
        std::cout << "this is k1" << k1 << std::endl;

        //auto inputhelper1 = dt / 2;
        //std::cout << "this is dt/2" << std::endl;
        //std::cout << inputhelper1 << std::endl;
        //std::cout << " " << std::endl;
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

    // TODO: double check the dimensions of X?
     torch::Tensor rollout(const torch::Tensor &state, const torch::Tensor &actions) const override
     {
         torch::Tensor X = torch::empty({ state.size(0), actions.size(1) }); // same # rows state, # cols action?
         auto curr_state = state.clone();
         for (int i = 0; i < actions.size(1); ++i) // not sure if this is the right dimension to loop by
         {
             torch::Tensor action = actions.index({ indexing::Ellipsis, i, indexing::Slice() });
             torch::Tensor next_state = predict(curr_state, action);
             std::cout << " rollout does a prediction of next state" << std::endl;
             X.select(1, i).copy_(next_state);
             curr_state = next_state.clone();
         }

         std::cout << " rollout is going to return" << std::endl;
         std::cout << X << std::endl;
         return X;
     }

    // think this implementation of rollout may be faster, if not then try using other one ajsidofjisd
    //torch::Tensor rollout(torch::Tensor state, torch::Tensor actions) const override
    //{
    //    std::vector<torch::Tensor> X;
    //    torch::Tensor curr_state = state;
    //    for (int t = 0; t < actions.size(-2); ++t) {
    //        auto action = actions.index({indexing::Ellipsis, t, indexing::Ellipsis});
    //        auto next_state = predict(curr_state, action);
    //        X.push_back(next_state);
    //        curr_state = next_state.clone();
    //    }
    //return torch::stack(X, -2);
    //}

    //torch::Tensor rollout(const torch::Tensor& state, const torch::Tensor& actions) const override
    //{
    //    // Initialize X with zeros
    //    torch::Tensor X = torch::zeros({ state.size(0), actions.size(1) });

    //    std::cout << "initialized X with zeros" << std::endl;

    //    // Initialize current state
    //    torch::Tensor curr_state = state;

    //    std::cout << "initialized curr_state" << std::endl;

    //    // Iterate through each action
    //    for (int i = 0; i < actions.size(1); ++i)
    //    {
    //        torch::Tensor action = actions.select(1, i);
    //        std::cout << "getting an action" << std::endl;

    //        torch::Tensor next_state = predict(curr_state, action);

    //        std::cout << "correctly performed prediction" << std::endl;

    //        // Update X with the next state
    //        X.select(1, i).copy_(next_state);

    //        std::cout << "updating X with next state" << std::endl;

    //        // Update current state
    //        curr_state = next_state.clone();
    //    }
    //    std::cout << "completed the rollout!" << std::endl;
    //    std::cout << X << std::endl;
    //    return X;
    //}

    //torch::Tensor rollout(const torch::Tensor& state, const torch::Tensor& actions) const override
    //{
    //    // Get shapes
    //    std::vector<int64_t> state_shape = state.sizes().vec();
    //    std::vector<int64_t> actions_shape = actions.sizes().vec();
    //    int64_t T = actions_shape.back(); // Get the size of the time dimension

    //    // Define the output tensor
    //    std::vector<int64_t> output_shape = state_shape;
    //    output_shape.insert(output_shape.end() - 1, T); // Insert T at the appropriate position
    //    torch::Tensor X = torch::empty(output_shape, state.options());

    //    std::cout << "it makes the output tensors" << std::endl;

    //    torch::Tensor curr_state = state.clone();
    //    std::cout << "it makes the clone" << std::endl;

    //    // Iterate over the time dimension
    //    for (int t = 0; t < T; ++t) {

    //        std::cout << "beginning of loop" << std::endl;

    //        // Extract action at time t
    //        torch::Tensor action = actions.index({ indexing::Ellipses, t, indexing::Slice() });

    //        std::cout << "extracts action" << std::endl;

    //        // Predict next state using the 'predict' function
    //        torch::Tensor next_state = predict(curr_state, action);

    //        std::cout << "prediction works" << std::endl;

    //        // Store next state in the output tensor
    //        X.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(), t, torch::indexing::Slice() }, next_state);

    //        std::cout << "store next state works" << std::endl;

    //        // Update current state
    //        curr_state = next_state.clone();
    //        
    //        std::cout << "curr_state updated" << std::endl;
    //    }

    //    return X;
    //}

    //TODO : is this returning a double? just want to double check
    torch::Tensor quat_to_yaw(const torch::Tensor &q)
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
    //torch::Tensor get_observations(const torch::Tensor &batch) const
    //{
    //    // torch::Tensor state = batch['state']; // can i index like this in c++?

    //    // // 

    //    // torch::Tensor x = state.index({torch::Slice(), torch::Slice(), torch::Slice(), 0}); // how do i do ... indexing in c++
    //    // torch::Tensor y = state.index({torch::Slice(), torch::Slice(), torch::Slice(), 1});
    //    // torch::Tensor q = state.slice(0, 3, 7); // ok this slicing makes no sense

    //    // this should be the proper way to index, assuming that namespace was done correctly
    //    auto state = batch.index({"state"});
    //    if (state.dim() == 1) {
    //        return get_observations(torch::indexing::dict_map(batch, [](torch::Tensor x){ return x.unsqueeze(0); })).squeeze();
    //    }

    //    auto x = state.index({indexing::Ellipsis, 0});
    //    auto y = state.index({indexing::Ellipsis, 1});
    //    auto q = state.index({indexing::Ellipsis, Slice(3, 7)});
    //    auto yaw = quat_to_yaw(q);
    //    return torch::stack({x, y, yaw}, -1);
    //}

    // is this function necessary to have
    //torch::Tensor get_actions(const torch::Tensor &batch) const
    //{
    //    return batch;
    //}

    

};

#endif