#ifndef COST_TERM_BASE_IS_INCLUDED
#define COST_TERM_BASE_IS_INCLUDED
// Base class for cost function terms. At a high-level, cost function terms need to be able to do the following:
//     1. Tell the high-level cost manager what data it needs to compute costs
//     2. Actually compute cost values given:
//         a. A [B1 x B2 x T x N] tensor of states
//         b. A [B1 x B2 x T x M] tensor of actions
// Note that we assume two batch dimensions as we often want to perform multiple sampling-based opts in parallel
class CostTerm
{
    public:
        CostTerm(){};
        virtual ~CostTerm();
        virtual void get_data_keys() = 0;
        virtual void cost() = 0;
};

#endif