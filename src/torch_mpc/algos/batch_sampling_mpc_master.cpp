#include "batch_sampling_mpc.h"
#include "../setup_mpc.h"
#include <filesystem>
#include <cmath>
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
    auto costmap_metadata_yaml = YAML::LoadFile(costmaps_dir.string());

    // Open data

    auto data_jit = torch::jit::load("C:/Users/anton/Documents/SPRING24/AEC/run_3/sample/sample.pth");
   
    auto data_tensor = data_jit.attr("data").toTensor();
    std::cout << data_tensor.sizes() << std::endl;
    // Open pos
    torch::jit::Module pos_jit = torch::jit::load(pos_dir.string());
    auto pos_tensor = pos_jit.attr("data").toTensor();
    std::cout << pos_tensor.sizes() << std::endl;
    // Open steer
    torch::jit::Module steer_jit = torch::jit::load(steer_dir.string());
    auto steer_tensor = steer_jit.attr("data").toTensor();
    std::cout << steer_tensor.sizes() << std::endl;
    // Open waypoints
    auto waypoints_yaml = YAML::LoadFile(waypoints_dir.string());

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

    int interval_data = std::ceil(static_cast<double>(len_pos) / len_data);
    int interval_steer = std::ceil(static_cast<double>(len_pos) / len_steer);
    int interval_waypoints = std::ceil(static_cast<double>(len_pos) / len_waypoints);
    int interval_metadatas = std::ceil(static_cast<double>(len_pos) / len_metadatas);


    auto mppi = setup_mpc(config); // returns a shared pointer to BatchSamplingMPC
    auto model = mppi->model; // returns a shared pointer to Model
    auto cfn = mppi->cost_function; // returns a shared pointer to CostFunction

    mppi->to(*device);

    // Testing can_compute_cost and inserting data
    std::vector<torch::Tensor> X;
    std::vector<torch::Tensor> U;
    std::vector<torch::Tensor> TRAJ;
    std::string dir_path = "C:/Users/anton/Documents/SPRING24/AEC/torch_mpc/src/torch_mpc/algos/algos_data";
    for(int i = 0; i < len_pos; ++i){
        auto current_pos = pos_tensor[i].to(*device);
        auto current_data = data_tensor[i / interval_data].to(*device);
        auto current_steer = steer_tensor[i / interval_steer].to(*device);
        auto current_waypoints = waypoints[i / interval_waypoints];
        auto current_metadata = metadatas[i / interval_metadatas];

        Values val;
        val.metadata = current_metadata;
        val.data = current_data;
        cfn->data.keys["local_costmap"] = val;

        Values val_waypoints;
        val_waypoints.data = current_waypoints;
        cfn->data.keys["waypoints"] = val_waypoints;

        Values val_goals;
        val_goals.data = goals[i / interval_waypoints];
        cfn->data.keys["goals"] = val_goals;

        torch::Tensor x0 = torch::empty({batch_size, 3}, torch::TensorOptions().device(*device));

        for (int i = 0; i < batch_size; i++) {
            x0[i][0] = current_pos[0];
            x0[i][1] = current_pos[1];
            x0[i][2] = current_steer[0]; // Assuming current_steer is a tensor with one element.
        }

        torch::Tensor x_list = torch::empty({batch_size, 3}, torch::TensorOptions().device(*device));
        torch::Tensor u_list = torch::empty({batch_size, 2}, torch::TensorOptions().device(*device));

        torch::Tensor x = x0.clone();
        for (int i = 0; i < 50; i++)
        {
            X.push_back(x.clone());
            auto [u, feasible] = mppi->get_control(x);
            U.push_back(u.clone());
            x = model->predict(x, u);
        }

        auto X_list = torch::stack(x_list, 1).to(*device);
        auto U_list = torch::stack(u_list, 1).to(*device);

        auto traj = model->rollout(x0, mppi->last_controls);

        X.push_back(X_list);
        U.push_back(U_list);
        TRAJ.push_back(traj);
    }


    auto X_tensor = torch::stack(X, 1).to(torch::kCPU);
    auto U_tensor = torch::stack(U, 1).to(torch::kCPU);
    auto TRAJ_tensor = torch::stack(TRAJ, 1).to(torch::kCPU);

    torch::save(X_tensor, dir_path+"X.pt");
    torch::save(U_tensor, dir_path+"U.pt");
    torch::save(TRAJ_tensor, dir_path+"TRAJ.pt");

    // std::cout << "Saved data" << std::endl;

    return 0;
}