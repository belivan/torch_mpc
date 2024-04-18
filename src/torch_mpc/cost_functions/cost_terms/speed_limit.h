#ifndef SPEED_LIMIT_IS_INCLUDED
#define SPEED_LIMIT_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "base.h"
#include <string>
#include <utility>

class SpeedLimit
{
    private:
        double target_speed;
        double max_speed;
        double sharpness;
        int speed_idx;
        torch::Device device;
    
    public:
        SpeedLimit(double target_speed = 4.0, double max_speed = 5.0, double sharpness = 10.0, int speed_idx = 3, const torch::Device& device = torch::kCPU)
    : target_speed(target_speed), max_speed(max_speed), sharpness(sharpness), speed_idx(speed_idx), device(device) {}
        std::vector<std::string> get_data_keys()
        {
            return {};
        }
        std::pair<torch::Tensor, torch::Tensor> cost(const torch::Tensor& states, const torch::Tensor& actions, 
                                                    const torch::Tensor& feasible, const std::unordered_map<std::string, torch::Tensor>& data)
        {
            torch::Tensor speed = states.index({"...", speed_idx}).abs();
            torch::Tensor cost = torch::zeros({states.size(0), states.size(1)}, torch::TensorOptions().device(device));

            torch::Tensor within_speed_limit = speed <= max_speed;
            torch::Tensor new_feasible = within_speed_limit.all(-1);

            cost = (sharpness * (speed - target_speed)).exp().mean(-1);

            return std::make_pair(cost, new_feasible);
        }

        SpeedLimit& to(const torch::Device& device)
        {
            this->device = device;
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& os, const SpeedLimit& sl) {
            os << "SpeedLimit";
            return os;
        }
};

#endif