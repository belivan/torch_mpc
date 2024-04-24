#include "batch_sampling_mpc.h"
#include "../setup_mpc.h"

int main() 
{
    std::string config_file = "/home/pearlfranz/aec/torch_mpc/configs/costmap_speedmap.yaml"; // specify the path to the config file

    YAML::Node config = YAML::LoadFile(config_file);
    
    // std::cout << "Loaded config" << std::endl;
 
    const std::string device_config = config["common"]["device"].as<std::string>();
    std::optional<torch::Device> device;
    if (device_config == "cuda")
    {device.emplace(torch::kCUDA, 0);}
    else if (device_config == "cpu")
    {device.emplace(torch::kCPU); }
    else {throw std::runtime_error("Unknown device " + device_config);}

    const int batch_size = config["common"]["B"].as<int>();

    // std::cout << "Loaded config" << std::endl;
    // std::cout << "Device: " << device_config << std::endl;
    // std::cout << "Batch size: " << batch_size << std::endl;

    // Load generated sampele data
    // torch::jit::Module tensors = torch::jit::load("/home/pearlfranz/aec/torch_mpc/src/torch_mpc/algos/sample.pth");
    // torch::Tensor pos = tensors.attr("pos").toTensor();
    // torch::Tensor steer = tensors.attr("steer").toTensor();
    // torch::Tensor data = tensors.attr("data").toTensor();
    // torch::Tensor waypoints = tensors.attr("waypoints").toTensor();

    // std::cout << "pos: " << pos << std::endl;
    // std::cout << "steer: " << steer << std::endl;
    // std::cout << "data: " << data.sizes() << std::endl;
    // std::cout << "waypoints: " << waypoints << std::endl;
    
    // Creating MPC

    auto mppi = setup_mpc(config); // returns a shared pointer to BatchSamplingMPC
    auto model = mppi->model; // returns a shared pointer to Model
    auto cfn = mppi->cost_function; // returns a shared pointer to CostFunction

    mppi->to(*device);

    // Testing can_compute_cost and inserting data
    Values val;

    // std::cout << "Check 0" << std::endl;
    // std::cout << cfn->can_compute_cost() << std::endl;
    // std::cout << "Expected: 0" << std::endl;

    // Simulate data loading
    torch::Tensor goal1 = torch::tensor({{250.0, 0.0}, 
                                        {600.0, 0.0}}, torch::TensorOptions().device(*device));
    torch::Tensor goal2 = torch::tensor({{300.0, 0.0}, 
                                        {700.0, 0.0}}, torch::TensorOptions().device(*device));
    torch::Tensor goal3 = torch::tensor({{350.0, 0.0}, 
                                        {800.0, 0.0}}, torch::TensorOptions().device(*device));

    auto goals= torch::stack({goal1, goal2, goal3}, 0);
    val.data = goals;
    
    cfn->data.keys["waypoints"] = val;

    // std::cout << "Check 1" << std::endl;
    // std::cout << cfn->can_compute_cost() << std::endl;
    // std::cout << "Expected: 0" << std::endl;

    Values val2 = val;
    cfn->data.keys["goals"] = val2;

    // std::cout << "Check 2" << std::endl;
    // std::cout << cfn->can_compute_cost() << std::endl;
    // std::cout << "Expected: 0" << std::endl;
    
    std::unordered_map<std::string, torch::Tensor> metadata;
    metadata["resolution"] = torch::tensor({2.5}, torch::TensorOptions().device(*device));
    metadata["width"] = torch::tensor({80.0}, torch::TensorOptions().device(*device));
    metadata["height"] = torch::tensor({80.0}, torch::TensorOptions().device(*device));
    metadata["origin"] = torch::tensor({-40.5, -40.5}, torch::TensorOptions().device(*device));
    metadata["length_x"] = torch::tensor({80.0}, torch::TensorOptions().device(*device));
    metadata["length_y"] = torch::tensor({80.0}, torch::TensorOptions().device(*device));

    Values val3 = val;
    val3.data = torch::zeros({1, 32, 32}, torch::TensorOptions().device(*device));
    val3.metadata = metadata;

    cfn->data.keys["local_costmap"] = val3;

    // std::cout << "Check 3" << std::endl;
    // std::cout << cfn->can_compute_cost() << std::endl;
    // std::cout << "Expected: 0" << std::endl;

    Values val4 = val3;
    cfn->data.keys["local_speedmap"] = val3;

    // std::cout << "Check 4" << std::endl;
    // std::cout << cfn->can_compute_cost() << std::endl;
    // std::cout << "Expected: 1" << std::endl;

    // std::cout << "Check 5 (Final)" << std::endl;
    // std::cout << "Plz don't crash" << std::endl;
    // END OF TESTING


    // std::cout << "Received MPC" << std::endl;
    // std::cout << mpc << std::endl; this won't print anything because the << operator is not defined for BatchSamplingMPC

    // Another script shows auto states = torch::zeros({3, 4, 100, 5});
    auto x = torch::zeros({batch_size, model->observation_space()}, torch::TensorOptions().device(*device));
    // std::cout << x.sizes() << std::endl;
    // auto x = torch::zeros({batch_size, 4, 100, 5}, torch::TensorOptions().device(*device));
    // x.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), 0}) = torch::linspace(0, 60, 100);

    // std::cout << "Created state" << std::endl;
    // std::cout << x << std::endl;

    std::vector<torch::Tensor> X; // X is the state
    std::vector<torch::Tensor> U; // U is the control input

    // std::cout << "Starting MPC" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 500; i++)
    for (int i = 0; i < 50; i++)
    {
        X.push_back(x.clone());
        auto [u, feasible] = mppi->get_control(x);
        // std::cout << "U: " << u.sizes() << std::endl;
        U.push_back(u.clone());
        x = model->predict(x, u);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // std::cout << "TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " milliseconds" << std::endl;

    auto X_tensor = torch::stack(X, 1).to(torch::kCPU);
    auto U_tensor = torch::stack(U, 1).to(torch::kCPU);

    // std::cout << "X: " << X_tensor.sizes() << std::endl;
    // std::cout << "U: " << U_tensor.sizes() << std::endl;

    // std::cout << "Rolling out with model" << std::endl;
    model->to(torch::kCPU);
    auto traj = model->rollout(x, mppi->last_controls).to(torch::kCPU);

    // std::cout << "TRAJ COST = " << std::endl;
    // std::cout << "X_tensor" << X_tensor.sizes() << std::endl;
    // std::cout << "U_tensor" << U_tensor.sizes() << std::endl;
    // std::cout << cfn->cost(X_tensor, U_tensor)<< std::endl;
    // auto result = cfn->cost(X_tensor, U_tensor);
    // std::cout << "TRAJ COST = " << result.first.sizes() << std::endl;
    // std::cout << "TRAJ FEASIBLE = " << result.second.sizes() << std::endl;

    auto du = torch::abs(U_tensor.slice(1,1) - U_tensor.slice(1,0,-1));

    // std::cout << "SMOOTHNESS = " << du.view({batch_size, -1}).mean(-1) << std::endl;

    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
    {
        auto u = mppi->get_control(x);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    // std::cout << "ITR TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " milliseconds" << std::endl;

    std::string dir_path = "/home/pearlfranz/aec/torch_mpc/src/torch_mpc/algos/algos_data/";
    torch::save(X_tensor, dir_path + "X.pt");
    torch::save(U_tensor, dir_path + "U.pt");
    torch::save(traj, dir_path + "traj.pt");

    // std::cout << "Saved data" << std::endl;

    return 0;
}



    // class TestCost
    // {   
    //     private:
    //         torch::Device device;

    //     public:
    //         TestCost()
    //         {
    //             device = torch::kCPU;
    //         }

    //         torch::Tensor cost(torch::Tensor traj, torch::Tensor controls){
    //             auto stage_cost = stage_cost(traj, controls);
    //             auto term_cost = term_cost(traj.index({"...", -1, torch::indexing::Slice()}));
    //             return stage_cost.sum(-1) + term_cost;
    //         }
    //         torch::Tensor stage_cost(const torch::Tensor& traj, const torch::Tensor& controls) {
    //             // drive fast in -x direction
    //             auto state_cost = traj.index({"...", 0}) + 10.0 * (traj.index({"...", 2}) - M_PI).abs();
    //             auto control_cost = torch::zeros_like(state_cost); // Assume control cost is simplified to zero tensor

    //             return state_cost + control_cost;
    //         }
    //         torch::Tensor term_cost(const torch::Tensor& tstates) {
    //             return tstates.index({"...", 1}) * 0.0;
    //         }
    //         TestCost& to(const torch::Device& device) {
    //             this->device = device;
    //             return *this;
    //         }
    // };