#ifndef GOAL_CONSTRAINT_IS_INCLUDED
#define GOAL_CONSTRAINT_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "base.h"
#include <string>
#include <utility>

class GoalConstraint
{
    private:
        double goal_radius;
        std::vector<std::string> goal_key;
        torch::Device device;

    public:
        GoalConstraint(double goal_radius = 2.0, const std::string& goal_key = "waypoints", const torch::Device& device = torch::kCPU)
    : goal_radius(goal_radius), goal_key(goal_key), device(device) {}

        std::vector<std::string> get_data_keys()
        {
            return goal_key;
        }

        std::pair<torch::Tensor, torch::Tensor> cost(const torch::Tensor& states, const torch::Tensor& actions,
                    const torch::Tensor& feasable, const torch::Tensor& data)
        {
            torch::Tensor cost = torch::zeros({states.size(0), states.size(1)},
                                            torch::TensorOptions().device(device));
            torch::Tensor new_feasible = torch::ones({states.size(0), states.size(1)},
                                                    torch::TensorOptions().dtype(torch::kBool).device(device));
            for (int bi = 0; bi < states.size(0); ++bi)
            {
                torch::Tensor bgoals = data.at(goal_key)[bi];
                torch::Tensor world_pos = states.index({bi, torch::indexing::Slice(), torch::indexing::Slice(None, 2)});
                torch::Tensor first_goal_dist = torch::norm(world_pos - bgoals[0], /*dim=*/-1);
                torch::Tensor traj_reached_goal = (first_goal_dist < goal_radius).any(/*dim=*/-1) & feasible.index({bi});
                if (traj_reached_goal.any().item<bool>())
                {
                    std::cout << "constraining!" << std::endl;
                    new_feasible.index_put_({bi}, traj_reached_goal);
                }
                else
                {
                    new_feasible.index_put_({bi}, feasible.index({bi}));
                }
            }
            return std::make_pair(cost, new_feasible);
        }

        GoalConstraint& to(const torch::Device& device)
        {
            this->device = device;
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& os, const GoalConstraint& gc) {
        os << "Goal Constraint";
        return os;
    }
};

#endif