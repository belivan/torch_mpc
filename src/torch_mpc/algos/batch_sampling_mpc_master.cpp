#include "batch_sampling_mpc.h"
#include "../setup_mpc.h"
#include <filesystem>
namespace fs = std::filesystem;

int main() 
{
    std::string config_file = "C:/Users/anton/Documents/SPRING24/AEC/torch_mpc/configs/costmap_speedmap.yaml"; // specify the path to the config file

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



    fs::path data_dir_path = "C:/Users/anton/Documents/SPRING24/AEC/run_3";
    fs::path costmaps_dir = data_dir_path / "costmaps/metadata.yaml";
    fs::path data_dir = data_dir_path / "data/data.pth";
    fs::path pos_dir = data_dir_path / "pos/pos.pth";
    fs::path steer_dir = data_dir_path / "steer/steer_data.pth";
    fs::path waypoints_dir = data_dir_path / "waypoints/forward.yaml";

    // Load generated sampele data
    // Open costmaps
    auto costmap_metadata_yaml = YAML::LoadFile(costmaps.string());

    // Open data

    auto data_jit = torch::jit::load("C:/Users/anton/Documents/SPRING24/AEC/run_3/sample/sample.pth");
   
    auto data_tensor = data_jit.attr("data").toTensor();
    std::cout << data_tensor.sizes() << std::endl;
    // Open pos
    torch::jit::Module pos_jit = torch::jit::load(pos.string());
    auto pos_tensor = pos_jit.attr("data").toTensor();
    std::cout << pos_tensor.sizes() << std::endl;
    // Open steer
    torch::jit::Module steer_jit = torch::jit::load(steer.string());
    auto steer_tensor = steer_jit.attr("data").toTensor();
    std::cout << steer_tensor.sizes() << std::endl;
    // Open waypoints
    auto waypoints_yaml = YAML::LoadFile(waypoints.string());

    std::cout << "Loaded data" << std::endl;
    // waypoints
    torch::Tensor waypoints = torch::Tensor();
    for(auto iter = waypoints_yaml["waypoints"].begin(); iter != waypoints_yaml["waypoints"].end(); ++iter)
    {
        auto sv = *iter;
        auto x = sv["pose"]["x"].as<double>();
        auto y = sv["pose"]["y"].as<double>();
        auto goal = torch::tensor({{x, y},{0.0, 0.0}}, torch::TensorOptions().device(*device));
        if (waypoints.numel() == 0){waypoints = goal;}
        else{waypoints = torch::cat({waypoints, goal}, 0);}
    }

    torch::Tensor goals = torch::clone(waypoints);
    // costmaps metadata
    std::vector<std::unordered_map<std::string, torch::Tensor>> metadatas;
    for(auto iter = costmap_metadata_yaml["costmaps"].begin(); iter != costmap_metadata_yaml["costmaps"].end(); ++iter)
    {
        auto mt = *iter;
        auto height = mt["height"].as<double>();
        bool first = true;
        double x;
        double y;
        for(auto oi = mt["origin"].begin(); iter != mt["origin"].end(); ++oi)
        {
            auto ox = *oi;
            if (first){first = false; x = ox.second.as<double>(); continue;}
            y = ox.second.as<double>();
        }
        auto origin = torch::tensor({x,y}, torch::TensorOptions().device(*device));
        auto resolution = torch::tensor({mt["resolution"].as<double>()}, torch::TensorOptions().device(*device));
        auto width = torch::tensor({mt["width"].as<double>()}, torch::TensorOptions().device(*device));
        auto height = torch::tensor({mt["height"].as<double>()}, torch::TensorOptions().device(*device));
        auto length_x = torch::clone(width);
        auto length_y = torch::clone(height);

        auto metadata_map = std::unordered_map<std::string, torch::Tensor>();
        metadata_map["resolution"] = resolution;
        metadata_map["width"] = width;
        metadata_map["height"] = height;
        metadata_map["origin"] = origin;
        metadata_map["length_x"] = length_x;
        metadata_map["length_y"] = length_y;
        metadatas.push_back(metadata_map);
    }

    auto len_pos = pos_tensor.size(0);
    auto len_data = data_tensor.size(0);
    auto len_steer = steer_tensor.size(0);
    auto len_waypoints = waypoints.size(0);
    auto len_metadatas = metadatas.size();  //same with goals















    // Creating MPC

    auto mppi = setup_mpc(config); // returns a shared pointer to BatchSamplingMPC
    auto model = mppi->model; // returns a shared pointer to Model
    auto cfn = mppi->cost_function; // returns a shared pointer to CostFunction

    mppi->to(*device);

    // Testing can_compute_cost and inserting data
    Values val;


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

    auto x = torch::zeros({batch_size, model->observation_space()}, torch::TensorOptions().device(*device));

    std::vector<torch::Tensor> X; // X is the state
    std::vector<torch::Tensor> U; // U is the control input

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 50; i++)
    {
        X.push_back(x.clone());
        auto [u, feasible] = mppi->get_control(x);
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