#include <torch/torch.h>
#include "base.h"
#include <unordered_map>

#ifndef TIP_OVER_COST_IS_INCLUDED
#define TIP_OVER_COST_IS_INCLUDED

namespace indexing = torch::indexing;

class TipOverCost : public CostTerm
{
private:
    float L;
    torch::Tensor alpha_del_steer, alpha_yaw_rate, max_del_steer, max_steer, max_del_steer_vel_range;
    int max_yaw_rate;
    torch::Device device;

public:

    TipOverCost(float max_del_steer, float max_steer, std::vector<float> max_del_steer_vel_range, float max_yaw_rate = -1, float alpha_del_steer = 30, float alpha_yaw_rate = 30, float L = 3.0, torch::Device device = torch::kCPU)
        : L(L), max_yaw_rate(max_yaw_rate), device(device)
    {
        this->alpha_del_steer = torch::clamp(torch::exp(torch::tensor(alpha_del_steer)), torch::tensor(1e-10));
        this->alpha_yaw_rate = torch::clamp(torch::exp(torch::tensor(alpha_yaw_rate)), torch::tensor(1e-10));

        this->max_del_steer.push_back(max_del_steer);
        this->max_steer.push_back(max_steer);
        this->max_del_steer_vel_range = max_del_steer_vel_range;
    }

    std::vector<std::string> get_data_keys() override
    {
        return {};
    }

    //def cost(self, states, actions, data,past_cost=None,cur_obs = None):
    torch::Tensor cost(torch::Tensor states, torch::Tensor actions, torch::Tensor data, torch::Tensor past_cost, void* cur_obs) override
    {
        torch::Tensor cost = torch::zeros({states.size(0), states.size(1)}, torch::TensorOptions().device(device));

        torch::Tensor world_speed = states.index({indexing::Ellipsis, 3}).abs();
        torch::Tensor future_steer = states.index({indexing::Ellipsis, 4});

        torch::Tensor cur_steer;
        if (cur_obs != nullptr) {
            if (auto dict_obs = dynamic_cast<std::unordered_map<std::string, torch::Tensor>*>(cur_obs)) {
                cur_steer = (*dict_obs)["new_state"].index({indexing::Ellipsis, -1});
            } else if (auto tensor_obs = static_cast<torch::Tensor*>(cur_obs)) {
                cur_steer = tensor_obs->index({indexing::Ellipsis, -1});
            } else {
                // Handle invalid pointer type
                // You can throw an exception or return an error tensor
                torch::Tensor bleh;
                return bleh;
            }
        }

        torch::Tensor init_change_in_steer = actions.index({indexing::Ellipsis, 0, -1}) - cur_steer;

        torch::Tensor max_del_steer = torch::zeros_like(init_change_in_steer);
        torch::Tensor max_steer = torch::zeros_like(actions.index({indexing::Ellipsis, -1}));
        for (int i = 0; i < max_del_steer_vel_range.size(0); ++i) {
            torch::Tensor mask;
            if (i == 0) {
                mask = world_speed < max_del_steer_vel_range[i];
            } else {
                mask = world_speed < max_del_steer_vel_range[i];
                mask = torch::logical_and(mask, world_speed > max_del_steer_vel_range[i - 1]);
            }
            max_del_steer.index_put_({mask.index({indexing::Ellipsis, 0})}, max_del_steer[i]);
            max_steer.index_put_({mask}, max_steer[i]);
        }

        torch::Tensor remaining_mask = max_del_steer == 0;
        max_del_steer.index_put_({remaining_mask}, max_del_steer[-1]);
        max_steer.index_put_({remaining_mask}, max_steer[-1]);

        cost += alpha_del_steer * (init_change_in_steer.abs() > max_del_steer);

        if (max_yaw_rate > 0) {
            torch::Tensor yaw_rate = world_speed * torch::tan(future_steer) / L;
            cost += torch::sum(alpha_yaw_rate * (yaw_rate.abs() > max_yaw_rate), -1);
        }

        cost += torch::sum(alpha_del_steer * (states.index({indexing::Ellipsis, -1}).abs() > max_steer), -1);

        return cost;
    }

    TipOverCost& to(torch::Device device) override
    {
        this->device = device;
        return *this;
    }

    std::string __repr__ () const override
    {
        return "Tip Over Cost";
    }

};

#endif