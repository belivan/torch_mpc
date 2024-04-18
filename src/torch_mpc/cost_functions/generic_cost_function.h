#ifndef GENERIC_COST_FUNCTION_IS_INCLUDED
#define GENERIC_COST_FUNCTION_IS_INCLUDED

#include <torch/torch.h>
#include "base.h"

class GenericCostFunction : public CostTerm // unsure if we also need to make the termfunctions?
{
private:
    std::vector<std::pair<float, TermFunctions>> cost_terms_;
    std::vector<torch::Tensor*> data_;
    torch::Device device_;

public:
    bool canComputeCost() const
    {
        for (auto &val : data_)
        {
            if(!val) // checking if nullptr?
            {
                return false;
            }
        }
        return true;
    }

    CostFunction to(torch::Device device) {
        device_ = device;
        for (auto &term : cost_terms_)
        {
            term.second->to(device);
        }
        return *this;
    }

};

#endif