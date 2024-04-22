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
        // std::cout << "is this where the code is going?" << std::endl;
        torch::Tensor cost = torch::zeros({ states.size(0), states.size(1) },
            torch::TensorOptions().device(device));
        torch::Tensor new_feasible = torch::ones({ states.size(0), states.size(1) },
            torch::TensorOptions().dtype(torch::kBool).device(device));
        // for-loop here because the goal array can be ragged

        // std::cout << "starting for loop euclidean" << std::endl;
        for (int bi = 0; bi < states.size(0); ++bi)
        {
            auto bgoals = utils::get_key_data_tensor(data, goal_key[0]).index({ bi });  // Double check this
            std::cout << "bgoals " << bgoals << std::endl;
            if (num_goals == -1)
            {
                num_goals = bgoals.size(0);
            }
            //std::cout << states.sizes() << "\nstates sizes\n\n\n" << std::endl;
            auto world_pos = states[bi].index({ Ellipsis, Slice(None, 2) });

            //std::cout << "world_pos " << world_pos << std::endl;

            // compute whether any trajs have reached the first goal
            for (int i = 0; i < num_goals; ++i)
            {
                //std::cout << "entering loop num_goals" << std::endl;
                //auto first_goal_dist = torch::norm(world_pos - bgoals[i], 2, -1);

                /*std::cout << "world_pos shape: " << world_pos.sizes() << std::endl;
                std::cout << "bgoals shape: " << bgoals.sizes() << std::endl;
                std::cout << "Index i: " << i << std::endl;*/

                //torch::Tensor goal_diff = world_pos - bgoals[i];
                //torch::Tensor goal_diff = world_pos - bgoals.index({ i });
                //std::cout << "goal_diff " << goal_diff << std::endl;
                //torch::Tensor first_goal_dist = torch::norm(goal_diff, 2, -1);
                //torch::Tensor first_goal_dist = torch::norm(goal_diff, torch::Tensor(), -1);
                auto first_goal_dist = torch::norm(world_pos - bgoals[i], 2, -1);
                //std::cout << "first_goal_dist " << first_goal_dist << std::endl;
                auto traj_reached_goal = torch::logical_and(torch::any(torch::lt(first_goal_dist, goal_radius), -1), feasible[bi][i]); //not sure adding [bi][i] makes a difference
                std::cout << "traj_reached_goal " << traj_reached_goal.sizes() << std::endl;

                if (i != (bgoals.size(0) - 1) && torch::any(traj_reached_goal).item<bool>())
                {
                    std::cout << "first" << std::endl;
                    cost[bi] += std::get<0>(torch::min(first_goal_dist, -1));
                    std::cout << "cost updated" << std::endl;
                    new_feasible = traj_reached_goal;
                    std::cout << "new feasible and cost update" << new_feasible.sizes() << std::endl;
                }
                else
                {
                    std::cout << "second" << std::endl;
                    cost[bi] += first_goal_dist.index({ Ellipsis, -1 });
                    break;
                }
                std::cout << "beyond the ifelse" << std::endl;
            }
        }

        std::cout << "returning euclidean" << std::endl;
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