#include "batch_sampling_mpc.h"
#include "../setup_mpc.h"
#include <filesystem>
#include <cmath>
#include <yaml-cpp/yaml.h>
// #include "fssimplewindow.h"

namespace fs = std::filesystem;

int main() 
{
    // set current path as file path
    fs::current_path(fs::path(__FILE__).parent_path());
    // now set config file path
    fs::current_path(fs::path("../../../configs/"));
    // check if the directory exists
    if (fs::exists(fs::path("../../../configs/")))
    {
        std::cout << "Directory does not exist" << std::endl;
        std::cout << "Dir: " << fs::current_path() << "\n";
        return 1;
    }

    std::string config_file = "costmap_speedmap.yaml"; // specify the path to the config file

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
    const int rollout_period = config["common"]["H"].as<int>();
    int num_steps = config["replay"]["steps"].as<int>();

    std::cout << "Loaded config" << std::endl;
    std::cout << "Device: " << device_config << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Rollout period: " << rollout_period << std::endl;
    std::cout << "Num steps: " << num_steps << std::endl;

    fs::current_path(fs::path("../../data/mppi_inputs/run_4/"));
    if(fs::exists(fs::path("../../data/mppi_inputs/run_4/")))
    {
        std::cout << "Directory does not exist" << std::endl;
        std::cout << "Dir: " << fs::current_path() << "\n";
        return 1;
    }
    std::string data_dir_path = "./";
    std::string costmaps_dir = data_dir_path + "costmaps/metadata.yaml";
    std::string data_dir = data_dir_path + "data/data.pth";
    std::string pos_dir = data_dir_path + "pos/pos_data.pth";
    std::string steer_dir = data_dir_path + "steer/steer_data.pth";
    std::string waypoints_dir = data_dir_path + "waypoints/forward.yaml";

    // Load generated sampele data
    // Open costmaps
    YAML::Node costmap_metadata_yaml = YAML::LoadFile(costmaps_dir);

    // Open data

    auto data_jit = torch::jit::load(data_dir);
   
    auto data_tensor = data_jit.attr("data").toTensor();
    std::cout << data_tensor.sizes() << std::endl;
    // Open pos
    torch::jit::Module pos_jit = torch::jit::load(pos_dir);
    auto pos_tensor = pos_jit.attr("data").toTensor();
    std::cout << pos_tensor.sizes() << std::endl;
    // Open steer
    torch::jit::Module steer_jit = torch::jit::load(steer_dir);
    auto steer_tensor = steer_jit.attr("data").toTensor();
    std::cout << steer_tensor.sizes() << std::endl;
    // Open waypoints
    auto waypoints_yaml = YAML::LoadFile(waypoints_dir);

    std::cout << "Loaded data" << std::endl;
    // waypoints
    torch::Tensor waypoints = torch::Tensor();
    for(auto iter = waypoints_yaml["waypoints"].begin(); iter != waypoints_yaml["waypoints"].end(); ++iter)
    {
        auto sv = *iter;
        auto x = sv["pose"]["x"].as<double>();
        auto y = sv["pose"]["y"].as<double>();
        auto goal = torch::tensor({{x, y},{0.0, 0.0}}, torch::TensorOptions().device(*device));
        // std::cout << "Goal: " << goal.sizes() << std::endl;
        if (waypoints.numel() == 0){waypoints = goal.unsqueeze(0);
        // std::cout << "Waypoints: " << waypoints.sizes() << std::endl;
        // std::cout << "here \n" << std::endl;
        }
        else{waypoints = torch::cat({waypoints, goal.unsqueeze(0)}, 0);}
    }
    // std::cout << "Waypoints: " << waypoints.sizes() << std::endl;

    // "Goals" not needed
    // torch::Tensor goals = torch::clone(waypoints);
    // costmaps metadata
    std::vector<std::unordered_map<std::string, torch::Tensor>> metadatas;
    for(auto iter = costmap_metadata_yaml["costmaps"].begin(); iter != costmap_metadata_yaml["costmaps"].end(); ++iter)
    {
        auto mt = *iter;
        auto origin = torch::tensor({mt["origin"]["x"].as<double>(), mt["origin"]["x"].as<double>()}, torch::TensorOptions().device(*device));
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

    std::cout << "Updating [  ] every:" << std::endl;
    std::cout << "Data: " << interval_data << std::endl;
    std::cout << "Steer: " << interval_steer << std::endl;
    std::cout << "Waypoints: " << interval_waypoints << std::endl;
    std::cout << "Metadatas: " << interval_metadatas << std::endl;


    auto mppi = setup_mpc(config); // returns a shared pointer to BatchSamplingMPC
    auto model = mppi->model; // returns a shared pointer to Model
    auto cfn = mppi->cost_function; // returns a shared pointer to CostFunction

    mppi->to(*device);

    std::vector<torch::Tensor> X;
    std::vector<torch::Tensor> U;
    std::vector<torch::Tensor> TRAJ;
    std::vector<torch::Tensor> COSTS;
    std::vector<torch::Tensor> X_TRUE;
    std::vector<torch::Tensor> GOALS;
    std::vector<torch::Tensor> COSTMAPS;

    fs::current_path(fs::path("../../../torch_mpc/src/torch_mpc/algos/algos_data/"));
    // check if the directory exists
    if (fs::exists(fs::path("../../../torch_mpc/src/torch_mpc/algos/algos_data/")))
    {
        std::cout << "Directory does not exist" << std::endl;
        std::cout << "Dir: " << fs::current_path() << "\n";
        return 1;
    }

    //FsOpenWindow(0,0,300,300,1);
    std::string dir_path = "./test3/";

    // temp
    // num_steps = 20;
    for(int i = 0; i < num_steps; ++i){
        std::cout << "Currenty sampling " << i << " / " << num_steps << " trajectories \n" << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();

        auto current_pos = pos_tensor[i].to(*device);
        auto current_data = data_tensor[i / interval_data].to(*device);
        auto current_steer = steer_tensor[i / interval_steer].to(*device);
        auto current_metadata = metadatas[i / interval_metadatas];
        for (auto& [key, value] : current_metadata)
        {value = value.to(*device);}

        // I want to select 2 goals/waypoints for each batch
        auto current_waypoints = waypoints.index({Slice(i / interval_waypoints, i / interval_waypoints + 2)}).to(*device);
        std::cout << "Current waypoints: " << current_waypoints.sizes() << std::endl;
        GOALS.push_back(current_waypoints);

        // std::cout << "Current pos: " << current_pos << std::endl;
        std::cout << "Current costmap data: " << current_data.sizes() << std::endl;
        COSTMAPS.push_back(current_data);
        // std::cout << "Current steer: " << current_steer << std::endl;
        // std::cout << "Current metadata: " << current_metadata["origin"] << std::endl;


        Values val;
        val.metadata = current_metadata;
        val.data = current_data;
        cfn->data.keys["local_costmap"] = val;

        Values val_waypoints;
        val_waypoints.data = current_waypoints;
        cfn->data.keys["waypoints"] = val_waypoints;

        // Values val_goals;
        // val_goals.data = goals.index({Slice(i / interval_waypoints, i / interval_waypoints + 3)}).to(*device);
        // cfn->data.keys["goals"] = val_goals;

        torch::Tensor x0 = torch::empty({batch_size, 3}, torch::TensorOptions().device(*device));

        for (int i = 0; i < batch_size; i++) {
            x0.index_put_({i, 0}, current_pos[0]);
            x0.index_put_({i, 1}, current_pos[1]);
            x0.index_put_({i, 2}, current_steer[0]);
        }
        std::cout << "x0: " << x0 << std::endl;

        std::vector<torch::Tensor> x_list;
        std::vector<torch::Tensor> u_list;

        torch::Tensor x = x0.clone();
        X_TRUE.push_back(x.clone());
        for (int i = 0; i < rollout_period; i++)
        {
            x_list.push_back(x.clone());
            auto [u, feasible] = mppi->get_control(x);
            u_list.push_back(u.clone());
            x = model->predict(x, u);
        }

        auto X_list = torch::stack(x_list, 1).to(*device);
        auto U_list = torch::stack(u_list, 1).to(*device);

        auto traj = model->rollout(x0, mppi->last_controls);

        X.push_back(X_list);
        U.push_back(U_list);
        TRAJ.push_back(traj);

        auto cost = std::get<0>(cfn->cost(X_list.unsqueeze(0), U_list.unsqueeze(0)));
        std::cout << "COST: " << cost << std::endl;
        COSTS.push_back(cost);

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "TOOK: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " milliseconds" << std::endl;
    }

    std::cout << "Finished sampling" << std::endl;
    auto X_tensor = torch::stack(X, 1).to(torch::kCPU);
    auto U_tensor = torch::stack(U, 1).to(torch::kCPU);
    auto TRAJ_tensor = torch::stack(TRAJ, 1).to(torch::kCPU);
    auto COSTS_tensor = torch::stack(COSTS, 0).to(torch::kCPU);
    auto X_TRUE_tensor = torch::stack(X_TRUE, 1).to(torch::kCPU);
    auto GOALS_tensor = torch::stack(GOALS, 0).to(torch::kCPU);
    auto COSTMAPS_tensor = torch::stack(COSTMAPS, 0).to(torch::kCPU);

    std::cout << "Saving data" << std::endl;
    std::cout << "X: " << X_tensor.sizes() << std::endl;
    std::cout << "U: " << U_tensor.sizes() << std::endl;
    std::cout << "TRAJ: " << TRAJ_tensor.sizes() << std::endl;
    std::cout << "COSTS: " << COSTS_tensor.sizes() << std::endl;
    std::cout << "X_TRUE: " << X_TRUE_tensor.sizes() << std::endl;
    std::cout << "GOALS: " << GOALS_tensor.sizes() << std::endl;
    std::cout << "COSTMAPS: " << COSTMAPS_tensor.sizes() << std::endl;

    std::cout << "Updated [ ] every:" << std::endl;
    std::cout << "Data: " << interval_data << std::endl;
    std::cout << "Steer: " << interval_steer << std::endl;
    std::cout << "Waypoints: " << interval_waypoints << std::endl;
    std::cout << "Metadatas: " << interval_metadatas << std::endl;

    torch::save(X_tensor, dir_path+"X.pt");
    torch::save(U_tensor, dir_path+"U.pt");
    torch::save(TRAJ_tensor, dir_path+"TRAJ.pt");
    torch::save(COSTS_tensor, dir_path+"COSTS.pt");
    torch::save(X_TRUE_tensor, dir_path+"X_TRUE.pt");
    torch::save(GOALS_tensor, dir_path+"GOALS.pt");
    torch::save(COSTMAPS_tensor, dir_path+"COSTMAPS.pt");

    std::cout << "Saved data" << std::endl;

    return 0;
}