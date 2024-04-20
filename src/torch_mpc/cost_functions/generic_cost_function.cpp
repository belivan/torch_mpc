#include "generic_cost_function.h"
#include "cost_terms/euclidean_distance_to_goal.h"
#include "cost_terms/footprint_costmap_projection.h"
#include "cost_terms/footprint_speedmap_projection.h"

int main() {
    std::vector<std::pair<double, std::shared_ptr<CostTerm>>> cost_terms = {
        {2.0, std::make_shared<EuclideanDistanceToGoal>()},
        {1.0, std::make_shared<FootprintSpeedmapProjection>()},
        {1.0, std::make_shared<FootprintCostmapProjection>()}
    };

    torch::Device device = torch::kCPU;

    CostFunction cfn(cost_terms, device);

    Data data;

    std::cout << "Check 0" << std::endl;
    std::cout << cfn.can_compute_cost() << std::endl;
    std::cout << "Expected: 0" << std::endl;

    // Simulate data loading
    torch::Tensor goal1 = torch::tensor({{5.0, 0.0}, {10.0, 0.0}});
    torch::Tensor goal2 = torch::tensor({{3.0, 0.0}, {4.0, 0.0}});
    torch::Tensor goal3 = torch::tensor({{6.0, 0.0}, {4.0, 0.0}});

    auto goals= torch::stack({goal1, goal2, goal3}, 0);
    data.data = goals;
    data.data = goals;
    
    cfn.data.keys["waypoints"] = data;

    std::cout << "Check 1" << std::endl;
    std::cout << cfn.can_compute_cost() << std::endl;
    std::cout << "Expected: 0" << std::endl;

    Data data2 = data;
    cfn.data.keys["goals"] = data2;

    std::cout << "Check 2" << std::endl;
    std::cout << cfn.can_compute_cost() << std::endl;
    std::cout << "Expected: 0" << std::endl;
    
    Metadata metadata;
    metadata.fields["resolution"] = torch::tensor({1.0, 0.5, 2.0});
    metadata.fields["width"] = torch::tensor({100., 50., 200.});
    metadata.fields["height"] = torch::tensor({100., 50., 200.});
    metadata.fields["origin"] = torch::tensor({{-50., -25., -100.}, {-50., -25., -100.}});
    metadata.fields["length_x"] = torch::tensor({100., 50., 200.});
    metadata.fields["length_y"] = torch::tensor({100., 50., 200.});

    Data data3 = data;
    data3.data = torch::zeros({3, 100, 100});
    data3.metadata = metadata;

    cfn.data.keys["local_costmap"] = data3;

    std::cout << "Check 3" << std::endl;
    std::cout << cfn.can_compute_cost() << std::endl;
    std::cout << "Expected: 0" << std::endl;

    Data data4 = data3;
    cfn.data.keys["local_speedmap"] = data3;

    std::cout << "Check 4" << std::endl;
    std::cout << cfn.can_compute_cost() << std::endl;
    std::cout << "Expected: 1" << std::endl;

    std::cout << "Check 5 (Final)" << std::endl;
    std::cout << "Plz don't crash" << std::endl;
    // Simulate states and actions
    auto states = torch::zeros({3, 4, 100, 5});
    states.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), 0}) = torch::linspace(0, 60, 100);

    auto actions = torch::zeros({3, 4, 100, 2});

    auto [costs, feasible] = cfn.cost(states, actions);
    std::cout << "Costs:\n" << costs << "\nFeasible:\n" << feasible << std::endl;

    return 0;
}