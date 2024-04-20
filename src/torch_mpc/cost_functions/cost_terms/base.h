#ifndef COST_TERM_BASE_IS_INCLUDED
#define COST_TERM_BASE_IS_INCLUDED
// Base class for cost function terms. At a high-level, cost function terms need to be able to do the following:
//     1. Tell the high-level cost manager what data it needs to compute costs
//     2. Actually compute cost values given:
//         a. A [B1 x B2 x T x N] tensor of states
//         b. A [B1 x B2 x T x M] tensor of actions
// Note that we assume two batch dimensions as we often want to perform multiple sampling-based opts in parallel
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>
class CostTerm
{
    public:
        CostTerm(){};
        virtual ~CostTerm();
        virtual std::vector<std::string> get_data_keys() const = 0;
        virtual std::pair<torch::Tensor, torch::Tensor> cost(
            const torch::Tensor& states, 
            const torch::Tensor& actions, 
            const torch::Tensor& feasible, 
            const std::unordered_map<std::string, std::variant<torch::Tensor,
                  std::unordered_map<std::string, std::variant<torch::Tensor,
                  std::unordered_map<std::string, torch::Tensor>>>>>& data) = 0;
        // virtual std::pair<torch::Tensor, torch::Tensor> cost(const torch::Tensor& states, const torch::Tensor& actions, 
        //                const torch::Tensor& feasible, const std::unordered_map<std::string, torch::Tensor>& data) = 0;
        virtual CostTerm& to(const torch::Device& device) = 0;
        friend std::ostream& operator<<(std::ostream& os, const CostTerm& term);
};

std::ostream& operator<<(std::ostream& os, const CostTerm& term) {
    os << "CostTerm";
    return os;
}

#endif // COST_TERM_BASE_IS_INCLUDED