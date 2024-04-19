#ifndef EUCLIDEAN_DISTANCE_TO_GOAL_IS_INCLUDED
#define EUCLIDEAN_DISTANCE_TO_GOAL_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "base.h"
#include <string>
#include <utility>

class EuclideanDistanceToGoal : public CostTerm
{
    private:
        double goal_radius;
        const std::vector<std::string> goal_key;
        const torch::Device device;
        unsigned int num_goals = 2;

    public:
        EuclideanDistanceToGoal(double goal_radius = 2.0, cosnt std::string& goal_key = "waypoints", 
                                const torch::Device& device = torch::kCPU) : goal_radius(goal_radius), goal_key(goal_key), device(device) {}
        ~EuclideanDistanceToGoal() = default;

        std::vector<std::string> get_data_keys() const override
        {
            return goal_key;
        }

        std::pair cost(const torch::Tensor& states, const torch::Tensor& actions,
                    const torch::Tensor& feasable, 
                    const const std::unordered_map<std::string, torch::Tensor>& data) override
        {
            torch::Tensor cost = torch.zeros({states.size(0), states.size(1)},
                                            torch::TensorOptions().device(device));
            torch::Tensor new_feasible = torch::ones({states.size(0), states.size(1)},
                                                    torch::TensorOptions().dtype(torch::kBool).device(device));
            for (int bi = 0; bi < states.size(0); bi++)
            {
                for (int t = 0; t < states.size(1); t++)
                {
                    auto bgoals = data[goal_key][bi];  // this means data is a dictionary.. change it
                    if (num_goals == -1)
                    {
                        num_goals = bgoals.size(0);
                    }
                    auto world_pos = states[bi].slice(2, 0, 2);

                    for (int i = 0; i < num_goals; i++)
                    {
                        auto first_goal_dist = torch::norm(world_pos - bgoals[i].unsqueeze(0).unqueeze(0), 2, -1);
                        auto traj_reached_goal = torch::any(first_goal_dist < goal_radius, 0) && feasable; // is feasible boolean?
                        if (i != (bgoals.size(0)-1) && torch::any(traj_reached_goal))
                        {
                            cost[bi] += std::get<0>(torch::min(first_goal_dist, 2));
                            new_feasible = traj_reached_goal;
                        }
                        else
                        {
                            cost[bi] += first_goal_dist.select(2, first_goal_dist.size(2)-1);
                            break;
                        }
                    }
                }
            }
            return std::make_pair(cost, new_feasible);
        }

        EuclideanDistanceToGoal& to(const torch::Device& device) override
        {
            this->device = device;
            return *this;
        }

        std::string __repr__ () const override
        {
            return "Euclidean DTG";
        }
};

#endif