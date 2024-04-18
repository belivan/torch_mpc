#include "generic_cost_function.h"

int main() {
    std::vector<std::pair<double, std::shared_ptr<CostTerm>>> cost_terms = {
        {2.0, std::make_shared<EuclideanDistanceToGoal>()},
        {1.0, std::make_shared<CostmapProjection>()}
    };

    CostFunction cfn(cost_terms);

    std::cout << cfn.can_compute_cost() << std::endl;

    // Simulate data loading
    // std::unordered_map<std::string, torch::Tensor> data;
    cf.data["goals"] = torch::tensor({{{5.0, 0.0},
                                    {10.0, 0.0}}, 
                                    {{3.0, 0.0}, 
                                    {4.0, 0.0}}, 
                                    {{6.0, 0.0}, 
                                    {4.0, 0.0}}});

    std::cout << cfn.can_compute_cost() << std::endl;

    cfn.data["costmap"] = torch::zeros({3, 100, 100});
    cfn.data["costmap"].index({Slice(), Slice(40, 60), Slice(60, None)}) = 10;
    cfn.data["costmap_metadata"] = {
        {"resolution", torch::tensor({1.0, 0.5, 2.0})},
        {"width", torch::tensor({100., 50., 200.})},
        {"height", torch::tensor({100., 50., 200.})},
        {"origin", torch::tensor({{-50., -50.}, {-25., -25.}, {-100., -100.}})}
    };

    // Simulate states and actions
    auto states = torch::zeros({3, 4, 100, 5});
    states.index({"...", 0, "...", 0}) = torch::linspace(0, 60, 100).unsqueeze(0).unsqueeze(0);

    auto actions = torch::zeros({3, 4, 100, 2});

    auto [costs, feasible] = cfn.cost(states, actions);
    std::cout << "Costs:\n" << costs << "\nFeasible:\n" << feasible << std::endl;

    return 0;
}