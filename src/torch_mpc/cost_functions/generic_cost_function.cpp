#include "generic_cost_function.h"
#include "cost_terms/euclidean_distance_to_goal.h"
#include "cost_terms/footprint_costmap_projection.h"
#include "cost_terms/footprint_speedmap_projection.h"

int main() {
    std::vector<std::pair<double, std::shared_ptr<CostTerm>>> cost_terms = {
        {2.0, std::make_shared<EuclideanDistanceToGoal>()},
        {1.0, std::make_shared<FootprintSpeedmapProjection>()}
        {1.0, std::make_shared<FootprintCostmapProjection>()}
    };

    torch::Device device = torch::kCPU;

    CostFunction cfn(cost_terms, device);

    std::cout << cfn.can_compute_cost() << std::endl;

    // Simulate data loading
    cf.data["waypoints"] = torch::tensor({{{5.0, 0.0},
                                    {10.0, 0.0}}, 
                                    {{3.0, 0.0}, 
                                    {4.0, 0.0}}, 
                                    {{6.0, 0.0}, 
                                    {4.0, 0.0}}});

    std::cout << cfn.can_compute_cost() << std::endl;

    std::unordered_map<std::string, std::variant<torch::Tensor, std::unordered_map<std::string, torch::Tensor>>> data_holder;
    std::unordered_map<std::string, torch::Tensor> data;
    std::unordered_map<std::string, torch::Tensor> metadata;
    cfn.data["local_costmap"] = data_holder{
        std::make_pair(
            "data", torch::zeros({3, 100, 100})
            ),
        std::make_pair(
            "metadata", metadata{
                std::make_pair(
                    {"resolution", torch::tensor({1.0, 0.5, 2.0})}
                    ),
                std::make_pair(
                    {"width", torch::tensor({100., 50., 200.})}
                ),
                std::make_pair(
                    {"height", torch::tensor({100., 50., 200.})}
                ),
                std::make_pair(
                    {"origin", torch::tensor({{-50., -50.}, {-25., -25.}, {-100., -100.}})}
                ),
                std::make_pair(
                    {"length_x", torch::tensor({100., 50., 200.})}
                ),
                std::make_pair(
                    {"length_y", torch::tensor({100., 50., 200.})}
                )
            }
            )
    };

    std::cout << cfn.can_compute_cost() << std::endl;
    cfn.data["local_speedmap"] = data_holder{
        std::make_pair(
            "data", torch::zeros({3, 100, 100})
            ),
        std::make_pair(
            "metadata", metadata{
                std::make_pair(
                    {"resolution", torch::tensor({1.0, 0.5, 2.0})}
                    ),
                std::make_pair(
                    {"width", torch::tensor({100., 50., 200.})}
                ),
                std::make_pair(
                    {"height", torch::tensor({100., 50., 200.})}
                ),
                std::make_pair(
                    {"origin", torch::tensor({{-50., -50.}, {-25., -25.}, {-100., -100.}})}
                ),
                std::make_pair(
                    {"length_x", torch::tensor({100., 50., 200.})}
                ),
                std::make_pair(
                    {"length_y", torch::tensor({100., 50., 200.})}
                )
            }
            )
    };
    std::cout << cfn.can_compute_cost() << std::endl;

    // Simulate states and actions
    auto states = torch::zeros({3, 4, 100, 5});
    states.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), 0}) = torch::linspace(0, 60, 100);

    auto actions = torch::zeros({3, 4, 100, 2});

    auto [costs, feasible] = cfn.cost(states, actions);
    std::cout << "Costs:\n" << costs << "\nFeasible:\n" << feasible << std::endl;

    return 0;
}