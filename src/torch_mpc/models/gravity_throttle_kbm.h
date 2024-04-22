#ifndef GRAVITY_THROTTLE_KBM_IS_INCLUDED
#define GRAVITY_THROTTLE_KBM_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <string>
#include <limits>
#include <algorithm>

class GravityThrottleKBM {
private:
    double L;
    double dt;
    std::string state_key;
    std::string pitch_key;
    std::string steer_key;
    torch::Device device;

    std::vector<double> u_lb;
    std::vector<double> u_ub;
    std::vector<double> x_lb;
    std::vector<double> x_ub;

    std::vector<double> steer_lim;
    double steer_rate_lim;
    std::vector<double> actuator_model;

    bool requires_grad;
    int num_params;

public:
    GravityThrottleKBM(const std::vector<double>& actuator_model, double L = 3.0,
                       const std::vector<double>& throttle_lim = {0.0, 1.0},
                       const std::vector<double>& steer_lim = {-0.52, 0.52},
                       double steer_rate_lim = 0.45, double dt = 0.1, 
                       const std::string& device_config = "cpu", 
                       bool requires_grad = false, 
                       const std::string& state_key = "state",
                       const std::string& pitch_key = "pitch",
                       const std::string& steer_key = "steer_angle") 
        : L(L), dt(dt), state_key(state_key), pitch_key(pitch_key), 
          steer_key(steer_key), device(torch::kCPU), 
          steer_lim(steer_lim), steer_rate_lim(steer_rate_lim), 
          actuator_model(actuator_model), requires_grad(requires_grad) {

        if (device_config == "cuda") {
            device = torch::Device(torch::kCUDA);
        } else if (device_config == "cpu") {
            device = torch::Device(torch::kCPU);
        }

        // Setup limits
        u_lb = {throttle_lim[0], steer_lim[0]};
        u_ub = {throttle_lim[1], steer_lim[1]};
        x_lb = std::vector<double>(5, -std::numeric_limits<double>::infinity());
        x_ub = std::vector<double>(5, std::numeric_limits<double>::infinity());

        for (auto& x : this->actuator_model) {
            x = std::exp(x);
        }

        num_params = actuator_model.size();
    }

    int64_t observation_space() {
        return x_lb.size();
    }
    int64_t action_space() {
        return u_lb.size();
    }
    // def update_parameters(self, parameters): // not implemented
    // def reset_parameters(self, requires_grad=None): // not implemented
    
    torch::Tensor dynamics(const torch::Tensor& state, const torch::Tensor& action) {
        auto parts = state.unbind(-1); // might work might not
        auto x = parts[0], y = parts[1], theta = parts[2], v = parts[3], delta = parts[4], pitch = parts[5];
        auto actions = action.unbind(-1);
        auto throttle_og = actions[0], delta_des_og = actions[1];

        auto v_actual = v;
        auto delta_actual = delta;
        auto sign_friction = v.sign();
        auto throttle_net = actuator_model[1] * throttle_og - actuator_model[2] * v - 
                            actuator_model[3] * sign_friction * pitch.cos() - actuator_model[5] * 
                            9.81 * pitch.sin();

        auto delta_des = actuator_model[0] * (delta_des_og - delta);
        auto center_mask = return_to_center(delta, delta_des_og);
        delta_des.index_put_({center_mask}, delta_des.index({center_mask}) * actuator_model[4]);
        delta_des = delta_des.clamp(-steer_rate_lim, steer_rate_lim);

        auto xd = v_actual * theta.cos() * pitch.cos();
        auto yd = v_actual * theta.sin() * pitch.cos();
        auto thd = v * delta_actual.tan() / L;

        auto vd = throttle_net;
        auto deltad = delta_des;

        auto below_steer_lim = delta.lt(steer_lim[0]).to(torch::kFloat32);
        auto above_steer_lim = delta.gt(steer_lim[1]).to(torch::kFloat32);

        deltad = deltad.clamp(0., 1e3) * below_steer_lim + deltad * (1. - below_steer_lim);
        deltad = deltad.clamp(-1e3, 0.) * above_steer_lim + deltad * (1. - above_steer_lim);

        auto pitchd = torch::zeros_like(pitch);

        return torch::stack({xd, yd, thd, vd, deltad, pitchd}, -1);
    }
    
    torch::Tensor return_to_center(const torch::Tensor& cur, const torch::Tensor& target) {
        // Calculate sign comparisons
        auto mask_0 = torch::logical_and(torch::sign(cur) >= 0, torch::sign(target) >= 0);
        auto mask_1 = torch::logical_and(torch::sign(cur) < 0, torch::sign(target) < 0);
        auto mask_2 = torch::logical_and(torch::sign(cur) >= 0, torch::sign(target) < 0);
        auto mask_3 = torch::logical_and(torch::sign(cur) < 0, torch::sign(target) >= 0);

        // Combine masks to form larger conditions
        auto mask_same = torch::logical_or(mask_0, mask_1);
        auto mask_diff = torch::logical_or(mask_2, mask_3);

        // Further combination based on absolute values comparison
        auto mask_same_back_to_0 = torch::logical_and(mask_same, torch::abs(cur) > torch::abs(target));

        // Final result combining all conditions
        return torch::logical_or(mask_same_back_to_0, mask_diff);
    }

    torch::Tensor predict(const torch::Tensor& state, const torch::Tensor& action) {
        auto k1 = dynamics(state, action);
        auto k2 = dynamics(state + 0.5 * dt * k1, action);
        return state + dt * k2;
    }

    torch::Tensor rollout(const torch::Tensor& state, const torch::Tensor& actions) {
        /*
        Expected shapes:
            state: [B1 x ... x Bn x xd]
            actions: [B1 x ... x Bn x T x ud]
            returns: [B1 x ... x Bn X T x xd]
            */
        std::vector<torch::Tensor> X;
        auto curr_state = state;

        int64_t T = actions.dim() - 2;
        for (int t = 0; t < T; ++t) {
            auto action = actions.select(T, t);
            auto next_state = predict(curr_state, action);
            X.push_back(next_state);
            curr_state = next_state.clone();
        }
        int64_t X_dim = X[0].dim() - 2;
        return torch::stack(X, /*dim=*/X_dim);
    }

    // def quat_to_yaw(self, quat): // not implemented

    // Leave to be implemented later
    // torch::Tensor get_observations(const torch::Tensor& batch) {
    //     auto state = batch.index({state_key});

    //     if (state.dim() == 1) {
    //         // If state is a 1D tensor, unsqueeze it at dimension 0, recurse, 
    //         // and then squeeze the result
    //         auto new_batch = batch.unsqueeze(0); // see dict_map below, but this might work
    //         return get_observations(new_batch).squeeze(0);
    //     }

    //     auto x = state.index({"...", 0});
    //     auto y = state.index({"...", 1});
    //     auto q = state.index({"...", Slice(3, 7)});
    //     auto yaw = quat_to_yaw(q);
    //     auto v = torch::norm(state.index({"...", Slice(7, 10)}), {}, -1);

    //     auto actual_direction = torch::atan2(state.index({"...", 8}), state.index({"...", 7}));
    //     auto delta = batch.index({steer_key}).index({"...", 0}) * (30.0 / 415.0) * (-M_PI / 180.0);
    //     auto pitch = -batch.index({pitch_key}).index({"...", 0});

    //     return torch::stack({x, y, yaw, v, delta, pitch}, -1);
    // }

    // def dict_map(d1, fn):
    // if isinstance(d1, dict):
    //     return {k:dict_map(v, fn) for k,v in d1.items()}
    // else:
    //     return fn(d1)


    // def get_actions(self, batch): // not implemented

    GravityThrottleKBM& to(torch::Device device) {
        this->device = device;
        return *this;
    }
};

#endif // GRAVITY_THROTTLE_KBM_IS_INCLUDED