#include "action_sampler.h"

int main()
{
    YAML::Node config = YAML::LoadFile("/home/anton/Desktop/SPRING24/AEC/torch_mpc/configs/costmap_speedmap.yaml");
    printf("Loaded config\n");

    const std::string device_config = config["common"]["device"].as<std::string>();
    
    std::optional<torch::Device> device;

    if (device_config == "cuda")
    {
        device.emplace(torch::kCUDA, 0);
    }
    else if (device_config == "cpu")
    {
        device.emplace(torch::kCPU);
    }
    else
    {
        throw std::runtime_error("Unknown device " + device_config);
        return 1;
    }

    const int B = config["common"]["B"].as<int>();
    const int H = config["common"]["H"].as<int>();
    const int M = config["common"]["M"].as<int>();
    
    
    std::unordered_map<std::string, std::unique_ptr<SamplingStrategy>> sampling_strategies;

    for(auto iter = config["sampling_strategies"]["strategies"].begin(); iter != config["sampling_strategies"]["strategies"].end(); ++iter)
    {
        auto sv = *iter;

        std::string type = sv["type"].as<std::string>();
        if (type == "UniformGaussian")
        {
            const int K = sv["args"]["K"].as<int>();
            
            const std::vector<double> scale = sv["args"]["scale"].as<std::vector<double>>();
            
            auto strategy = std::make_unique<UniformGaussian>(scale, B, K, H, M, *device);
            sampling_strategies.emplace(sv["label"].as<std::string>(), std::move(strategy));
        }
        else if (type == "ActionLibrary")
        {
            const int K = sv["args"]["K"].as<int>();
            
            std::string path = sv["args"]["path"].as<std::string>();
        
            auto strategy = std::make_unique<ActionLibrary>(path, B, K, H, M, *device);
            sampling_strategies.emplace(sv["label"].as<std::string>(), std::move(strategy));
        }
        else if (type == "GaussianWalk")
        {
            const int K = sv["args"]["K"].as<int>();
            
            const std::vector<double> scale = sv["args"]["scale"].as<std::vector<double>>();
        
            std::unordered_map<std::string, std::variant<std::string, std::vector<double>>> initial_distribution;

            initial_distribution["type"] = sv["args"]["initial_distribution"]["type"].as<std::string>();
            initial_distribution["scale"] = sv["args"]["initial_distribution"]["scale"].as<std::vector<double>>();

            const std::vector<double> alpha = sv["args"]["alpha"].as<std::vector<double>>();

            auto strategy = std::make_unique<GaussianWalk>(initial_distribution, scale, alpha, B, K, H, M, *device);
            sampling_strategies.emplace(sv["label"].as<std::string>(), std::move(strategy));
            
        }
        else
        {
            throw std::runtime_error("Unknown sampling strategy " + type);
        }
    }

    ActionSampler action_sampler(sampling_strategies);
    printf("Using device: %s\n", action_sampler.deviceToString(*device).c_str());
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(*device);
    const torch::Tensor u_nominal = torch::stack({
        torch::linspace(0.0, 1.0, 50, options),
        torch::linspace(-0.5, -0.2, 50, options)
    }, 1).unsqueeze(0);

    const torch::Tensor u_lb = torch::tensor({0., -0.52}, *device).view({1, 2});
    const torch::Tensor u_ub = torch::tensor({1., 0.52}, *device).view({1, 2});

    auto t1 = std::chrono::high_resolution_clock::now();
    
    auto samples = action_sampler.sample_dict(u_nominal, u_lb, u_ub); //std::unordered_map<std::string, torch::Tensor> 

    auto t2 = std::chrono::high_resolution_clock::now();

    u_nominal.to(torch::kCPU);
    u_lb.to(torch::kCPU);
    u_ub.to(torch::kCPU);

    std::string dir_path = "/home/anton/Desktop/SPRING24/AEC/torch_mpc/src/torch_mpc/action_sampling/sampling_data/";
    torch::save(u_nominal, dir_path + "u_nominal.pt");
    torch::save(u_lb, dir_path + "u_lb.pt");
    torch::save(u_ub, dir_path + "u_ub.pt");

    for(const auto& [k, v] : samples)
    {
        v.to(torch::kCPU);
        torch::save(v, dir_path + k + ".pt");
    }

    std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms to sample and save outputs \n";
    // 4 ms without saving
    // 42 ms with saving
    // Python script 0.0855s or 85.5ms
    return 0;
}