#ifndef BATCH_SAMPLING_MPC_IS_INCLUDED
#define BATCH_SAMPLING_MPC_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <string>
#include <stdexcept>
#include <optional>
#include <chrono>  // check if this is needed
#include <variant>
#include <cmath>

#include "../models/base.h"
#include "../cost_functions/generic_cost_function.h"
#include "../update_rules/mppi.h"
#include "../action_sampling/action_sampler.h"
class BatchSamplingMPC
{
    /*
    Implementation of a relatively generic sampling-based MPC algorithm (e.g. CEM, MPPI, etc.)
    We can use this implementation to instantiate a number of algorithms cleanly
    */

    private:
        // std::shared_ptr<Model> model;
        // std::shared_ptr<CostFunction> cost_function;
        // std::shared_ptr<MPPI> update_rule;
        // std::shared_ptr<ActionSampler> action_sampler;

        int B;
        int H;
        int K;
        int M;
        int n;
        int m;
        torch::Device device = torch::kCPU;

        torch::Tensor last_states;
        // torch::Tensor last_controls;
        torch::Tensor noisy_controls;
        torch::Tensor noisy_states;
        torch::Tensor costs;
        torch::Tensor last_cost;
        torch::Tensor last_weights;
        torch::Tensor u_lb;
        torch::Tensor u_ub;

    public:
        // make them private if needed, needed for testing script
        std::shared_ptr<Model> model;
        std::shared_ptr<CostFunction> cost_function;
        std::shared_ptr<MPPI> update_rule;
        std::shared_ptr<ActionSampler> action_sampler;

        torch::Tensor last_controls;

        BatchSamplingMPC(std::shared_ptr<Model> model, std::shared_ptr<CostFunction> cost_function, 
                    std::shared_ptr<ActionSampler> action_sampler, std::shared_ptr<MPPI> update_rule)
        : model(std::move(model)), cost_function(std::move(cost_function)), 
        update_rule(std::move(update_rule)), action_sampler(std::move(action_sampler))
        {
            /*
            Args:
            model: The model to optimize over
            cost_fn: The cost function to optimize
            action_sampler: The strategy for sampling actions
            update_rule: The update rule to get best seq from samples
            */

            // might be an issue if empty
            this->B = this->action_sampler->B.value();
            this->H = this->action_sampler->H.value();
            this->K = this->action_sampler->K;
            this->device = this->action_sampler->device.value();
            n = this->model->observation_space(); //.size(0); // only works for gravity throttle kbm
            m = this->model->action_space(); //.size(0); // only works for gravity throttle kbm
            setup_variables();
        }

        void setup_variables()
        {
            // Setup all the variables necessary to run mpc.
            last_states = torch::zeros({B, H, n}, torch::TensorOptions().device(device));
            last_controls = torch::zeros({B, H, m}, torch::TensorOptions().device(device));
            noisy_controls = torch::zeros({B, K, H, m}, torch::TensorOptions().device(device));
            noisy_states = torch::zeros({B, K, H, n}, torch::TensorOptions().device(device));
            costs = torch::randn({B, K}, torch::TensorOptions().device(device));
            last_cost = torch::randn({B}, torch::TensorOptions().device(device));
            last_weights = torch::zeros({B, K}, torch::TensorOptions().device(device));
            // verify model's params
            u_lb = torch::tensor(this->model->u_lb, torch::TensorOptions().device(device).dtype(torch::kFloat)).view({1, m}).repeat({B, 1});
            u_ub = torch::tensor(this->model->u_ub, torch::TensorOptions().device(device).dtype(torch::kFloat)).view({1, m}).repeat({B, 1});
        }

        void reset()
        {
            setup_variables();
        }

        torch::Tensor rollout_model(const torch::Tensor& obs, const std::optional<torch::Tensor>& extra_states, const torch::Tensor& noisy_controls)
        {
            auto states_unsqueeze = obs.unsqueeze(1).expand({-1, K, -1});
            torch::Tensor trajs;
            // if (extra_states) // determine what to do with extra_states
            // {
            //     auto extra_state_unsqueezed = extra_states->unsqueeze(1).expand({-1, K, -1});
            //     trajs = model->rollout(states_unsqueeze, noisy_controls, extra_state_unsqueezed);
            // }
            // else
            // {
            trajs = model->rollout(states_unsqueeze, noisy_controls);  // probably should pass all params
            // }
            
            return trajs;
        }

        std::pair<torch::Tensor, torch::Tensor> get_control(const torch::Tensor& obs, bool step = true, const std::optional<torch::Tensor>& extra_states = std::nullopt)
        {
            /*
            Returns action u at current state obs
            Args:
                x: 
                    Current state. Expects tensor of size self.model.state_dim()
                step:
                    optionally, don't step the actions if step=False
            Returns:
                u: 
                    Action from MPPI controller. Returns Tensor(self.m)
                cost:
                    Cost from executing the optimal action sequence. Float.

            Batching convention is [B x K x T x {m/n}], where B = batchdim, K=sampledim, 
            T=timedim, m/n = state/action dim
        */
            // Simulate K rollouts and get costs for each rollout
            auto noisy_controls = action_sampler->sample(last_controls, u_lb, u_ub);
            auto trajs = rollout_model(obs, extra_states, noisy_controls);
            // std::cout << "trajs" << trajs.sizes() << std::endl;
            // std::cout << "noisy_controls" << noisy_controls.sizes() << std::endl;
            auto [costs, feasible] = cost_function->cost(trajs, noisy_controls);

            auto not_feasible = feasible.logical_not();
            costs = costs + not_feasible.toType(torch::kDouble) * 1e8;  // penalize infeasible trajs // I AM DOING SOME DOUBLE STUFF HERE 

            auto [optimal_controls, sampling_weights] = update_rule->update(noisy_controls, costs); // implement update

            // Return first action in sequence and update self.last_controls
            last_controls = optimal_controls;
            // std::cout << "last_controls" << last_controls.sizes() << std::endl;

            auto best_cost_idx = torch::argmin(costs, 1);
            // std::cout << "best_cost_idx" << best_cost_idx.sizes() << std::endl;

            last_states = noisy_states.index({torch::arange(B), best_cost_idx});
            // std::cout << "last_states" << last_states.sizes() << std::endl;
            last_cost = costs.index({torch::arange(B), best_cost_idx});
            // std::cout << "last_states" << last_cost.sizes() << std::endl;
            noisy_states = trajs;
            this->noisy_controls = noisy_controls;
            this->costs = costs;
            last_weights = sampling_weights;

            torch::Tensor u = last_controls.select(1, 0);

            if (step)
            {
                this->step();
            }

            // std::cout << "returning u" << std::endl;

            return {u, torch::any(feasible, 1)};
        }

        void step()
        {
            // Updates self.last_controls to warm start next action sequence.

            // Shift controls by one and Initialize last control to be the same as the last in the sequence
            last_controls = torch::cat({last_controls.narrow(1, 1, H-1), last_controls.narrow(1, H-1, 1)}, 1);
        }

        BatchSamplingMPC& to(const torch::Device& new_device)
        {
            device = new_device;
            u_lb = u_lb.to(device);
            u_ub = u_ub.to(device);
            last_controls = last_controls.to(device);
            noisy_controls = noisy_controls.to(device);
            noisy_states = noisy_states.to(device);
            costs = costs.to(device);
            last_cost = last_cost.to(device);
            last_weights = last_weights.to(device);

            model->to(device);
            cost_function->to(device);
            action_sampler->to(device);

            return *this;
        }
};      

#endif // BATCH_SAMPLING_MPC_IS_INCLUDED