#ifndef EUCLIDEAN_DISTANCE_TO_GOAL_IS_INCLUDED
#define EUCLIDEAN_DISTANCE_TO_GOAL_IS_INCLUDED

#include <torch/torch.h>
#include <cmath>
#include "base.h"
#include <utility>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include "utils.h"

using namespace torch::indexing;

class EuclideanDistanceToGoal : public CostTerm
{
private:
    double goal_radius;
    std::vector<std::string> goal_key;
    torch::Device device;
    int num_goals = 2;

public:
    EuclideanDistanceToGoal(const torch::Device& device = torch::kCPU,
        double goal_radius = 2.0,
        const std::vector<std::string>& goal_key = { "waypoints" })
        : goal_radius(goal_radius), goal_key(goal_key), device(device) {}
    // ~EuclideanDistanceToGoal() = default;

    std::vector<std::string> get_data_keys() const override
    {
        return goal_key;
    }

    std::pair<torch::Tensor, torch::Tensor> cost(
        const torch::Tensor& states,
        const torch::Tensor& actions,
        const torch::Tensor& feasible,
        const CostKeyDataHolder& data) override
    {
        torch::Tensor cost = torch::zeros({ states.size(0), states.size(1) },
            torch::TensorOptions().device(device));
        torch::Tensor new_feasible = torch::ones({ states.size(0), states.size(1) },
            torch::TensorOptions().dtype(torch::kBool).device(device));

        for (int bi = 0; bi < states.size(0); ++bi)
        {
            auto bgoals = utils::get_key_data_tensor(data, goal_key[0]); // .index({ bi });  // Double check this
            if (num_goals == -1)
            {
                num_goals = bgoals.size(0);
            }
            auto world_pos = states[bi].index({ Ellipsis, Slice(None, 2) });

            // compute whether any trajs have reached the first goal
            for (int i = 0; i < num_goals; ++i)
            {
                auto first_goal_dist = torch::norm(world_pos - bgoals[i][0], 2, -1);
                auto traj_reached_goal = torch::logical_and(torch::any(torch::lt(first_goal_dist, goal_radius), -1), feasible[bi]);

                if (i != (bgoals.size(0) - 1) && torch::any(traj_reached_goal).item<bool>())
                {
                    cost[bi] += std::get<0>(torch::min(first_goal_dist, -1));
                    new_feasible = traj_reached_goal;
                }
                else
                {
                    cost[bi] += first_goal_dist.index({ Ellipsis, -1 });
                    break;
                }
            }
        }

        // std::cout << "Returning Euclidean" << std::endl;
        return { cost, new_feasible };
    }

    EuclideanDistanceToGoal& to(const torch::Device& device) override
    {
        this->device = device;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const EuclideanDistanceToGoal& edtg);
};

std::ostream& operator<<(std::ostream& os, const EuclideanDistanceToGoal& edtg)
{
    os << "EuclideanDistanceToGoal(goal_radius=" << edtg.goal_radius << ", goal_key=" << edtg.goal_key << ", device=" << edtg.device << ")";
    return os;
}

#endif // EUCLIDEAN_DISTANCE_TO_GOAL_IS_INCLUDED