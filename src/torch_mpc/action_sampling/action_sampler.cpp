#include "action_sampler.h"

int main()
{
    YAML::Node config = YAML::LoadFile("/home/anton/Desktop/SPRING24/AEC/torch_mpc/configs/costmap_speedmap.yaml");
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
    const int dt = config["common"]["dt"].as<int>();
    
    std::unordered_map<std::string, std::unique_ptr<SamplingStrategy>> sampling_strategies;

    for(auto iter = config["sampling_strategies"]["strategies"].begin(); iter != config["sampling_strategies"]["strategies"].end(); ++iter)
    {
        auto sv = iter->second;

        std::string type = sv["type"].as<std::string>();
        const std::vector<double> scale = sv["args"]["scale"].as<std::vector<double>>();
        if (type == "UniformGaussian")
        {
            const int K = sv["args"]["K"].as<int>();

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
    
    const torch::Tensor u_nominal = torch::stack({
        torch::linspace(0.0, 1.0, 50),
        torch::linspace(-0.5, -0.2, 50)
    }, 1).unsqueeze(0);

    u_nominal.to(*device);

    const torch::Tensor u_lb = torch::tensor({0., -0.52}, *device).view({1, 2});
    const torch::Tensor u_ub = torch::tensor({1., 0.52}, *device).view({1, 2});

    auto t1 = std::chrono::high_resolution_clock::now();
    auto samples = action_sampler.sample_dict(u_nominal, u_lb, u_ub);
    auto t2 = std::chrono::high_resolution_clock::now();

    u_nominal.to(torch::kCPU);
    u_lb.to(torch::kCPU);
    u_ub.to(torch::kCPU);

    for(const auto& [k, v] : samples)
    {
        v.to(torch::kCPU);
    }

    std::cout << "took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/1000 << "s to sample\n";

    return 0;
}