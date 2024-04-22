#include "batch_sampling_mpc.h"
#include "../setup_mpc.h"

int main() 
{
    std::string config_file = "config.yaml"; // specify the path to the config file

    YAML::Node config = YAML::LoadFile(config_file);

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
 
    const std::string device_config = config["common"]["device"].as<std::string>();
    std::optional<torch::Device> device;
    if (device_config == "cuda")
    {device.emplace(torch::kCUDA, 0);}
    else if (device_config == "cpu")
    {device.emplace(torch::kCPU); }
    else {throw std::runtime_error("Unknown device " + device_config);}

    const int batch_size = config["common"]["B"].as<int>();
    
    // auto cfn = std::make_shared<TestCost>()->to(*device);
    

    // Creating ActionSampler
    // std::unordered_map<std::string, std::unique_ptr<SamplingStrategy>> sampling_strategies;

    // for(auto iter = config["sampling_strategies"]["strategies"].begin(); iter != config["sampling_strategies"]["strategies"].end(); ++iter)
    // {
    //     auto sv = *iter;

    //     std::string type = sv["type"].as<std::string>();
    //     if (type == "UniformGaussian")
    //     {
    //         const int K = sv["args"]["K"].as<int>();
            
    //         const std::vector<double> scale = sv["args"]["scale"].as<std::vector<double>>();
            
    //         auto strategy = std::make_unique<UniformGaussian>(scale, B, K, H, M, *device);
    //         sampling_strategies.emplace(sv["label"].as<std::string>(), std::move(strategy));
    //     }
    //     else if (type == "ActionLibrary")
    //     {
    //         const int K = sv["args"]["K"].as<int>();
            
    //         std::string path = sv["args"]["path"].as<std::string>();
        
    //         auto strategy = std::make_unique<ActionLibrary>(path, B, K, H, M, *device);
    //         sampling_strategies.emplace(sv["label"].as<std::string>(), std::move(strategy));
    //     }
    //     else if (type == "GaussianWalk")
    //     {
    //         const int K = sv["args"]["K"].as<int>();
            
    //         const std::vector<double> scale = sv["args"]["scale"].as<std::vector<double>>();
        
    //         std::unordered_map<std::string, std::variant<std::string, std::vector<double>>> initial_distribution;

    //         initial_distribution["type"] = sv["args"]["initial_distribution"]["type"].as<std::string>();
    //         initial_distribution["scale"] = sv["args"]["initial_distribution"]["scale"].as<std::vector<double>>();

    //         const std::vector<double> alpha = sv["args"]["alpha"].as<std::vector<double>>();

    //         auto strategy = std::make_unique<GaussianWalk>(initial_distribution, scale, alpha, B, K, H, M, *device);
    //         sampling_strategies.emplace(sv["label"].as<std::string>(), std::move(strategy));
            
    //     }
    //     else
    //     {
    //         throw std::runtime_error("Unknown sampling strategy " + type);
    //     }
    // }

    // // ActionSampler action_sampler(sampling_strategies);
    // auto action_sampler = std::make_shared<ActionSampler>(sampling_strategies);

    // Creating model
    // auto model = std::make_shared<KBM>(3.0, 0., 1., 0.3, 0.1, *device); 

    // // Creating update rule
    // auto update_rule = std::make_shared<MPPI>(config["update_rule"]["args"]["temperature"].as<double>());
    
    // Creating MPC
    // auto mppi = std::make_unique<BatchSamplingMPC>(model, cfn, mppi, action_sampler, update_rule);

    auto mppi = setup_mpc(config); // returns a unique pointer to BatchSamplingMPC
    auto model = mppi->model; // returns a shared pointer to Model
    auto cfn = mppi->cost_function; // returns a shared pointer to CostFunction

    auto x = torch::zeros({batch_size, model->observation_space()}, torch::Options(*device));

    std::vector<torch::Tensor> X; // X is the state
    std::vector<torch::Tensor> U; // U is the control input

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 500; i++)
    {
        X.push_back(x.clone());
        auto u = mppi->get_control(x);
        U.push_back(u.clone());
        x = model->predict(x, u);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " milliseconds" << std::endl;

    auto X_tensor = torch::stack(X, 1).to(torch::kCPU);
    auto U_tensor = torch::stack(U, 1).to(torch::kCPU);

    auto traj = model->rollout(X_tensor, mppi->last_controls).to(torch::kCPU);

    std::cout << "TRAJ COST = " << cfn->cost(X_tensor, U_tensor) << std::endl;

    auto du = torch::abs(U_tensor.slice(1,1) - U_tensor.slice(1,0,-1));

    std::cout << "SMOOTHNESS = " << du.view({batch_size, -1})mean(-1) << std::endl;

    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
    {
        auto u = mppi->get_control(x);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " milliseconds" << std::endl;

    std::string dir_path = "/home/anton/Desktop/SPRING24/AEC/torch_mpc/src/torch_mpc/algos/algos_data/";
    torch::save(X_tensor, dir_path + "X.pt");
    torch::save(U_tensor, dir_path + "U.pt");
    torch::save(traj, dir_path + "traj.pt");

    std::cout << "Saved data" << std::endl;

    return 0;
}
