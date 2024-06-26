#ifndef SETUP_MPC_IS_INCLUDED
#define SETUP_MPC_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <string>
#include <stdexcept>
#include <optional>
#include <chrono>
#include <variant>
#include <yaml-cpp/yaml.h>

// sampling strategies
#include "action_sampling/sampling_strategies/uniform_gaussian.h"
#include "action_sampling/sampling_strategies/gaussian_walk.h"
#include "action_sampling/sampling_strategies/action_library.h"
#include "action_sampling/action_sampler.h"

// cost functions
#include "cost_functions/cost_terms/base.h"
#include "cost_functions/cost_terms/footprint_costmap_projection.h"
#include "cost_functions/cost_terms/footprint_speedmap_projection.h"
#include "cost_functions/cost_terms/euclidean_distance_to_goal.h"
#include "cost_functions/generic_cost_function.h"

// models
#include "models/base.h"
#include "models/kbm.h"
#include "models/gravity_throttle_kbm.h"
#include "models/actuator_delay.h"


// update rules
#include "update_rules/mppi.h"

// mpc
#include "algos/batch_sampling_mpc.h"


std::shared_ptr<BatchSamplingMPC> setup_mpc(YAML::Node config)
{
    // call this function to set up an MPC instance from the config yaml
    const std::string device_config = config["common"]["device"].as<std::string>();
    // std::cout << "Loaded config at MPC Helper" << std::endl;

    std::optional<torch::Device> device;

    if (device_config == "cuda")
    {device.emplace(torch::kCUDA, 0);}
    else if (device_config == "cpu")
    {device.emplace(torch::kCPU); }
    else {throw std::runtime_error("Unknown device " + device_config);}

    const int B = config["common"]["B"].as<int>();
    const int H = config["common"]["H"].as<int>();
    const int M = config["common"]["M"].as<int>();
    const double dt = config["common"]["dt"].as<double>();
    
    // setup model
    // std::cout << "Setting up model" << std::endl;

    std::shared_ptr<Model> model;
    if (config["model"]["type"].as<std::string>() == "KBM")
    {
        const double L = config["model"]["args"]["L"].as<double>();

        const std::vector<double> throttle_lim = config["model"]["args"]["throttle_lim"].as<std::vector<double>>();
        const double min_throttle = throttle_lim[0];
        const double max_throttle = throttle_lim[1];

        const double max_steer = config["model"]["args"]["max_steer"].as<double>();
        
        model = std::make_shared<KBM>(L, min_throttle, max_throttle, 
                                            max_steer, dt, *device);
        // std::cout << "Created KBM" << std::endl;
    }
    // Not implemented yet
    // else if (config["model"]["type"].as<std::string>() == "GravityThrottleKBM")
    // {
    //     const double L = config["model"]["args"]["L"].as<double>();
    //     const std::vector<double> throttle_lim = config["model"]["args"]["throttle_lim"].as<std::vector<double>>();
    //     const std::vector<double> steer_lim = config["model"]["args"]["steer_lim"].as<std::vector<double>>();
    //     const double steer_rate_lim = config["model"]["args"]["steer_rate_lim"].as<double>();
    //     const std::vector<double> actuator_model = config["model"]["args"]["actuator_model"].as<std::vector<double>>();
        
    //     auto model = std::make_unique<GravityThrottleKBM>(actuator_model, L, 
    //                                                         throttle_lim, steer_lim, 
    //                                                         steer_rate_lim, dt, device_config); // using device_config here
    // }
    else
    {
        throw std::runtime_error("Unknown model type " + config["model"]["type"].as<std::string>());
    }

    // Not implemented yet
    // setup actuator delay
    // if (config["model"]["actuator_delay"] && 
    //    !config["model"]["actuator_delay"].IsNull() && 
    //     config["model"]["actuator_delay"].size()>0) 
    // {auto model = std::make_unique<ActuatorDelay>(model, config["model"]["actuator_delay"].as<std::vector<double>>());}

    // setup sampler
    // std::cout << "Setting up action sampler" << std::endl;
    std::unordered_map<std::string, std::shared_ptr<SamplingStrategy>> sampling_strategies;
    for(auto iter = config["sampling_strategies"]["strategies"].begin(); iter != config["sampling_strategies"]["strategies"].end(); ++iter)
    {
        // std::cout << "Setting up sampling strategy" << std::endl;
        auto sv = *iter;

        std::string type = sv["type"].as<std::string>();
        if (type == "UniformGaussian")
        {
            // std::cout << "Setting up UniformGaussian" << std::endl;
            const int K = sv["args"]["K"].as<int>();
            
            const std::vector<double> scale = sv["args"]["scale"].as<std::vector<double>>();
            
            auto strategy = std::make_shared<UniformGaussian>(scale, B, K, H, M, *device);
            sampling_strategies.emplace(sv["label"].as<std::string>(), std::move(strategy));
        }
        else if (type == "ActionLibrary")
        {
            // std::cout << "Setting up ActionLibrary" << std::endl;
            const int K = sv["args"]["K"].as<int>();
            
            std::string path = sv["args"]["path"].as<std::string>();
        
            auto strategy = std::make_shared<ActionLibrary>(path, B, K, H, M, *device);
            sampling_strategies.emplace(sv["label"].as<std::string>(), std::move(strategy));
        }
        else if (type == "GaussianWalk")
        {
            // std::cout << "Setting up GaussianWalk" << std::endl;
            const int K = sv["args"]["K"].as<int>();
            
            const std::vector<double> scale = sv["args"]["scale"].as<std::vector<double>>();
        
            std::unordered_map<std::string, std::variant<std::string, std::vector<double>>> initial_distribution;

            initial_distribution["type"] = sv["args"]["initial_distribution"]["type"].as<std::string>();
            initial_distribution["scale"] = sv["args"]["initial_distribution"]["scale"].as<std::vector<double>>();

            const std::vector<double> alpha = sv["args"]["alpha"].as<std::vector<double>>();

            auto strategy = std::make_shared<GaussianWalk>(initial_distribution, scale, alpha, B, K, H, M, *device);
            sampling_strategies.emplace(sv["label"].as<std::string>(), std::move(strategy));
        }
        else
        {
            throw std::runtime_error("Unknown sampling strategy " + type);
        }
    }

    // init action sampler
    auto action_sampler = std::make_shared<ActionSampler>(sampling_strategies);
    // ActionSampler action_sampler(sampling_strategies);

    // setup cost function
    // std::cout << "Setting up cost function" << std::endl;
    std::vector<std::pair<double, std::shared_ptr<CostTerm>>> terms;
    for(auto iter = config["cost_function"]["terms"].begin(); iter != config["cost_function"]["terms"].end(); ++iter)
    {
        auto term = *iter;
        double weight = term["weight"].as<double>();
        std::string type = term["type"].as<std::string>();

        // check if args is not empty
        auto params = YAML::Node();
        if (term["args"] && 
           !term["args"].IsNull() && 
            term["args"].size()>0) {
                std::cout << "Setting up params" << std::endl;
                params = term["args"];}

        if (type == "EuclideanDistanceToGoal")
        {
            // std::cout << "Setting up EuclideanDistanceToGoal" << std::endl;
            terms.push_back({weight, std::make_shared<EuclideanDistanceToGoal>(*device)});
        }
        else if (type == "FootprintSpeedmapProjection" && !params.IsNull() && params.size()>0)
        {
            // std::cout << "Setting up FootprintSpeedmapProjection" << std::endl;
            auto length = params["length"].as<double>();
            auto width = params["width"].as<double>();
            auto length_offset = params["length_offset"].as<double>();
            auto width_offset = params["width_offset"].as<double>();
            auto nl = params["nl"].as<int>();
            auto nw = params["nw"].as<int>();

            terms.push_back({weight, std::make_shared<FootprintSpeedmapProjection>(length, width,  nl, nw,
                                                                                    length_offset, width_offset, 
                                                                                    *device)});
        }
        else if (type == "FootprintCostmapProjection" && !params.IsNull() && params.size()>0)
        {
            // std::cout << "Setting up FootprintCostmapProjection" << std::endl;
            auto length = params["length"].as<double>();
            auto width = params["width"].as<double>();
            auto length_offset = params["length_offset"].as<double>();
            auto width_offset = params["width_offset"].as<double>();
            // auto cost_thresh = params["cost_thresh"].as<double>(); does not exist
            
            if (!params["nl"] || !params["nw"])
            {
                terms.push_back({weight, std::make_shared<FootprintCostmapProjection>(length, width, 
                                                                                 length_offset, width_offset, 
                                                                                 *device)});
                continue;
            }
            auto nl = params["nl"].as<int>();
            auto nw = params["nw"].as<int>();

            terms.push_back({weight, std::make_shared<FootprintCostmapProjection>(length, width, 
                                                                                 length_offset, width_offset, 
                                                                                 *device, nl, nw)});
        }
        else
        {
            throw std::runtime_error("Unknown cost term " + type);
        }
    }

    // CostFunction cost_function(terms, *device);
    auto cost_fn = std::make_shared<CostFunction>(terms, *device);

    // setup update rules
    // std::cout << "Setting up update rule" << std::endl;
    std::shared_ptr<MPPI> update_rule;
    if (config["update_rule"]["type"].as<std::string>() == "MPPI")
    {
        // MPPI update_rule(config["update_rule"]["args"]["temperature"].as<double>());
        update_rule = std::make_shared<MPPI>(config["update_rule"]["args"]["temperature"].as<double>());
    }
    else 
    {
        throw std::runtime_error("Unknown update rule " + config["update_rule"]["type"].as<std::string>());
    }

    // setup algo
    // BatchSamplingMPC algo(model, cost_fn, action_sampler, update_rule);
    // std::cout << "Setting up MPC" << std::endl;
    auto algo = std::make_shared<BatchSamplingMPC>(model, cost_fn, action_sampler, update_rule);
    return algo;
};


#endif // SETUP_MPC_IS_INCLUDED
