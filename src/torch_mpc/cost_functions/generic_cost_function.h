#ifndef GENERIC_COST_FUNCTION_IS_INCLUDED
#define GENERIC_COST_FUNCTION_IS_INCLUDED

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <functional>
#include "cost_terms/base.h"
#include <variant>

class CostFunction {
/*
High level cost-term aggregator that all MPC should be using to get the costs of trajectories

In order to handle constraints properly, all cost terms must produce a feasible flag, in
    addition to a cost. We can then check if all trajs are infeasible and do something about it.

Note that this feasible matrix is meant to be equivalent to allowing cost terms to prune out trajectories,
but maintaining the batch shape.
*/
private:
    std::vector<std::shared_ptr<CostTerm>> cost_terms;
    std::vector<double> cost_weights;
    torch::Device device;

public:
    std::unordered_map<std::string, std::variant<torch::Tensor,  // make mutable?
    std::unordered_map<std::string, torch::Tensor>>> data;

    CostFunction(const std::vector<std::pair<double, 
                std::shared_ptr<CostTerm>>>& terms, 
                const torch::Device& device = torch::kCPU)
    : device(device) {
        /*
         Args:
            cost_terms: a list of (weight, term) tuples that comprise the cost function
            data_timeout: time after which 
        */
        for (const auto& term : terms) {
            cost_weights.push_back(term.first);
            cost_terms.push_back(term.second);
        }
        for (const auto& key : get_data_keys()) { // double check double key function
            data[key] = torch::Tensor();  // Initialize with empty tensors
        }
    }

    std::pair<torch::Tensor, torch::Tensor> cost(const torch::Tensor& states, 
                                                const torch::Tensor& actions) {
        /*
        Produce costs from states, actions
        Args:
            states: a [B1 x B2 x T x N] tensor of states
            actions: a [B x B2 x T x M] tensor of actions
        */
        auto costs = torch::zeros({states.size(0), states.size(1)}, states.options());
        auto feasible = torch::ones({states.size(0), states.size(1)}, torch::kBool).to(device);

        for (size_t i = 0; i < cost_terms.size(); ++i) {
            auto [new_cost, new_feasible] = cost_terms[i]->cost(states, actions, feasible, data);
            costs += cost_weights[i] * new_cost;
            feasible = feasible.logical_and(new_feasible);
        }
        return {costs, feasible};
    }

    std::vector<std::string> get_data_keys() const {
        std::set<std::string> keys;
        for (const auto& term : cost_terms) {
            auto term_keys = term->get_data_keys();
            keys.insert(term_keys.begin(), term_keys.end());
        }
        return std::vector<std::string>(keys.begin(), keys.end());
    }

    bool can_compute_cost() const {
        for (const auto& v : data.values()) {
            if (v.numel() == 0) {
                return false;
            }
        }
        return true;
    }

    void to(const std::string& device_type) {
        device = torch::Device(device_type);
        for (auto& term : cost_terms) {
            term->to(device);  // Assuming CostTerm also supports a to(device) method
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const CostFunction& cf) {
        os << "Cost Function containing:\n";
        for (size_t i = 0; i < cf.cost_terms.size(); ++i) {
            os << "\t" << cf.cost_weights[i] << ":\t" << *cf.cost_terms[i] << "\n";
        }
        return os;
    }
};

#endif // COST_FUNCTION_H