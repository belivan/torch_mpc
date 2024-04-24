#include "generic_cost_function.h"
#include "cost_terms/euclidean_distance_to_goal.h"
#include "cost_terms/footprint_costmap_projection.h"
#include "cost_terms/footprint_speedmap_projection.h"

int main() {

    // This script might need adjustment to work properly
    // See what inputs are required for the cost function classes
    std::vector<std::pair<double, std::shared_ptr<CostTerm>>> cost_terms = {
       {2.0, std::make_shared<EuclideanDistanceToGoal>()},
       // {1.0, std::make_shared<FootprintSpeedmapProjection>()},
       // {1.0, std::make_shared<FootprintCostmapProjection>()}
    };

    torch::Device device = torch::kCPU;  // try torch::kCUDA later

    CostFunction cfn(cost_terms, device);

    Values val;

    std::cout << "Check 0" << std::endl;
    std::cout << cfn.can_compute_cost() << std::endl;
    std::cout << "Expected: 0" << std::endl;

    // Simulate data loading
    torch::Tensor goal1 = torch::tensor({{5.0, 0.0}, {10.0, 0.0}});
    torch::Tensor goal2 = torch::tensor({{3.0, 0.0}, {4.0, 0.0}});
    torch::Tensor goal3 = torch::tensor({{6.0, 0.0}, {4.0, 0.0}});

    auto goals= torch::stack({goal1, goal2, goal3}, 0);
    val.data = goals;
    
    cfn.data.keys["waypoints"] = val;

    std::cout << "Check 1" << std::endl;
    std::cout << cfn.can_compute_cost() << std::endl;
    std::cout << "Expected: 0" << std::endl;

    Values val2 = val;
    cfn.data.keys["goals"] = val2;

    std::cout << "Check 2" << std::endl;
    std::cout << cfn.can_compute_cost() << std::endl;
    std::cout << "Expected: 0" << std::endl;
    
    std::unordered_map<std::string, torch::Tensor> metadata;
    metadata["resolution"] = torch::tensor({1.0, 0.5, 2.0});
    metadata["width"] = torch::tensor({100., 50., 200.});
    metadata["height"] = torch::tensor({100., 50., 200.});
    metadata["origin"] = torch::tensor({{-50., -50.}, {-25., -25.}, {-100., -100.}});
    metadata["length_x"] = torch::tensor({100., 50., 200.}); //might neeed adjustment
    metadata["length_y"] = torch::tensor({100., 50., 200.}); //might neeed adjustment

    Values val3 = val;
    val3.data = torch::zeros({1, 100, 100});
    val3.metadata = metadata;

    cfn.data.keys["local_costmap"] = val3;

    std::cout << "Check 3" << std::endl;
    std::cout << cfn.can_compute_cost() << std::endl;
    std::cout << "Expected: 0" << std::endl;

    Values val4 = val3;
    cfn.data.keys["local_speedmap"] = val3;

    std::cout << "Check 4" << std::endl;
    std::cout << cfn.can_compute_cost() << std::endl;
    std::cout << "Expected: 1" << std::endl;

    std::cout << "Check 5 (Final)" << std::endl;
    std::cout << "Plz don't crash" << std::endl;
    // Simulate states and actions
    auto states = torch::zeros({3, 4, 100, 5});
    states.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), 0}) = torch::linspace(0, 60, 100);

    std::cout << "the states are created" << std::endl;

    auto actions = torch::zeros({3, 4, 100, 2});

    std::cout << "the actions are created" << std::endl;

    auto [costs, feasible] = cfn.cost(states, actions);
    std::cout << "Costs:\n" << costs << "\nFeasible:\n" << feasible << std::endl;

    return 0;
}