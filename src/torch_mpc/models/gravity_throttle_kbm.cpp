#include <yaml-cpp/yaml.h>
#include "gravity_throttle_kbm.h"

int main()
{
    std::string config_file = "config.yaml"; // path to config file

    YAML::Node config = YAML::LoadFile(config_file);
    std::cout << "Loaded config" << std::endl;

    auto mpc = setup_mpc(config); // returns a unique pointer to a BatchSamplingMPC instance
    std::cout << "Setup MPC (BatchSamplingMPC)" << std::endl;

    auto model = mpc->model; // returns a shared pointer to a GravityThrottleKBM instance
    std::cout << "Got model (GravityThrottleKBM)" << std::endl;

    auto x0 = torch::zeros({6}); // initial state
    auto U = torch::zeros({50, 2}); // initial control sequence
    U.slice(1, 0, 1) = 1;  // throttle
    U.slice(1, 1, 2) = 0.15; // steer

    auto X1 = model->rollout(x0, U); // rollout the model

    x0.index_put_({-1}, -0.2)
    auto X2 = model->rollout(x0, U); // rollout the model

    x0.index_put_({-1}, 0.2)
    auto X3 = model->rollout(x0, U); // rollout the model

    std::string dir_path = "/home/anton/Desktop/SPRING24/AEC/torch_mpc/src/torch_mpc/models/models_data/";
    torch::save(X1, dir_path + "X1.pt");
    torch::save(X2, dir_path + "X2.pt");
    torch::save(X3, dir_path + "X3.pt");

    std::cout << "Saved data" << std::endl;

    return 0;
}