#ifndef UTILS_IS_INCLUDED
#define UTILS_IS_INCLUDED

#include <torch/torch.h>
#include "base.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <utility>
#include <unordered_map>
#include <exception>
#include <variant>

namespace utils
{
    torch::Tensor move_to_local_frame(const torch::Tensor& traj, 
                                    int xidx = 0, int yidx = 1, int thidx = 2) {
            /*
            Transform a trajectory into its local frame. I.e., translate by -(x, y),
            then rotate by -th. These values will be taken from the initial state
            of each trajectory. Note that this only handles positions (i.e. assume
            velocity is scalar/in body frame)

        Args:
            traj: A [B x K x T x 2] Tensor of trajectories
            xidx: The index of the x coordinate in state
            yidx: The index of the y coordinate in state
            thidx: The index of the th coordinate in state
            */
        auto ops = traj.select(-2, 0).index_select(-1, torch::tensor({xidx, yidx}, torch::kLong))
                    .view({traj.size(0), traj.size(1), 1, 2});
        auto oths = traj.select(-2, 0).select(-1, thidx).view({traj.size(0), traj.size(1), 1, 1});

        // translate
        auto pos_out = traj.index_select(-1, torch::tensor({xidx, yidx}, torch::dtype(torch::kLong))) - ops;

        // rotate
        auto th_out = traj.select(-1, thidx) - oths.squeeze(-1); // added squeeze

        auto cos_th = torch::cos(oths);
        auto sin_th = torch::sin(oths);
        auto R = torch::stack({torch::stack({cos_th, -sin_th}, -1),
                            torch::stack({sin_th, cos_th}, -1)}, -1);

        pos_out = torch::matmul(R, pos_out.unsqueeze(-1)); // [B x K x T x 2] check this

        std::vector<int64_t> pos_out_new_size(pos_out.sizes().begin(), pos_out.sizes().end() - 1);
        pos_out = pos_out.view(pos_out_new_size);

        auto traj_out = traj.clone();
        traj_out.index_put_({torch::indexing::Slice(), xidx}, pos_out.select(-1, 0));
        traj_out.index_put_({torch::indexing::Slice(), yidx}, pos_out.select(-1, 1));
        traj_out.index_put_({torch::indexing::Slice(), thidx}, th_out.select(-1, 0));
        return traj_out;
    }

    std::pair<torch::Tensor, torch::Tensor> world_to_grid(const torch::Tensor& world_pos, 
                                const std::unordered_map<std::string, torch::Tensor>& metadata) {
        /*
        Converts the world position (x,y) into indices that can be used to access the costmap.
    
    Args:
        world_pos:
            Tensor(B, K, T, 2) representing the world position being queried in the costmap.
        metadata:
            res, length, width (B,)
            origin (B, 2)

    Returns:
        grid_pos:
            Tensor(B, K, T, 2) representing the indices in the grid that correspond to the world position
    */
        auto res = metadata.at("resolution");
        auto nx = (metadata.at("length_x") / res).to(torch::kLong);
        auto ny = (metadata.at("length_y") / res).to(torch::kLong);
        auto ox = metadata.at("origin").select(-1,0);
        auto oy = metadata.at("origin").select(-1,1);
        
        // helper function
        auto ones_tensor = torch::ones(world_pos.dim() - 2).to(torch::kLong);
        // end helper function

        std::vector<int64_t> trailing_dims(ones_tensor.data_ptr<int64_t>(), ones_tensor.data_ptr<int64_t>() + ones_tensor.numel());

        // helper function
        std::vector<int64_t> new_shape = {-1};
        new_shape.insert(new_shape.end(), trailing_dims.begin(), trailing_dims.end());
        // end helper function

        auto gx = (world_pos.select(-1, 0) - ox.view(new_shape)) / res.view(new_shape);
        auto gy = (world_pos.select(-1, 1) - oy.view(new_shape)) / res.view(new_shape);

        auto grid_pos = torch::stack({gx, gy}, -1).to(torch::kLong);
        auto invalid_mask = (grid_pos.select(-1, 0) < 0) |
                            (grid_pos.select(-1, 1) < 0) |
                            (grid_pos.select(-1, 0) >= nx.view(new_shape)) |
                            (grid_pos.select(-1, 1) >= ny.view(new_shape));

        return {grid_pos, invalid_mask};
    }   

    torch::Tensor value_iteration(const torch::Tensor& costmap, 
                            const std::unordered_map<std::string, torch::Tensor>& metadata, 
                            const torch::Tensor& goals, 
                            double tol = 1e-4, double gamma = 1.0, int max_itrs = 1000) {
    /*
    Perform (batched) value iteration on a costmap
    Args:
        costmap: Tensor (B x W x H) costmap to perform value iteration over
        metadata: map metadata
        goals: Tensor (B x 2) of goal positions for each costmap
    */
        int B = costmap.size(0);
        auto res = metadata.at("resolution");
        auto nx = (metadata.at("height") / res).to(torch::kLong)[0].item<int64_t>();
        auto ny = (metadata.at("width") / res).to(torch::kLong)[0].item<int64_t>();
        
        // setup
        auto V = torch::full({B, nx+2, ny+2}, 1e10, costmap.options());
        auto R = torch::full({B, nx+2, ny+2}, 1e10, costmap.options());
        R.slice(1, 1, -1).slice(2, 1, -1) = costmap;

        // load in goal point
        // goals are 3D tensor with 2D goal tensors
        auto goal_coords = goals.view({B, 1, 1, 2});

        auto goal_grid_pos = world_to_grid(goal_coords, metadata).first.view({B, 2}).index_select(-1, torch::tensor({1, 0}, torch::kLong));
        R.index_put_({torch::arange(B), goal_grid_pos.index({torch::indexing::Slice(), 0}) + 1, 
                                        goal_grid_pos.index({torch::indexing::Slice(), 1}) + 1}, 0);

        // perform value iteration
        for (int i = 0; i < nx + ny; ++i) {
            auto Rsa = torch::stack({costmap,
                                    costmap,
                                    costmap,
                                    costmap,
                                    costmap}, 1); // [B x 5 x W x H]

            auto batch = torch::arange(B, torch::kLong);
            //auto row = goal_grid_pos.index({"...", 0}) + 1;
            //auto col = goal_grid_pos.index({"...", 1}) + 1;
            auto row = goal_grid_pos.index({torch::indexing::Ellipsis, 0 }) + 1;
            auto col = goal_grid_pos.index({ torch::indexing::Ellipsis, 1 }) + 1;
            auto indices = torch::stack({batch, row, col}, 1);
            // handle terminal state
            V.index_put_({indices.index({torch::indexing::Slice(), 0}), 
                            indices.index({torch::indexing::Slice(), 1}), 
                            indices.index({torch::indexing::Slice(), 2})}, 0);

            auto V_stay = V.slice(2, 1, -1).slice(3, 1, -1);
            auto V_up = V.slice(2, 2, torch::indexing::None).slice(3, 1, -1);
            auto V_down = V.slice(2, 0, -2).slice(3, 1, -1);
            auto V_left = V.slice(2, 1, -1).slice(3, 2, torch::indexing::None);
            auto V_right = V.slice(2, 1, -1).slice(3, 0, -2);

            auto Vs_next = torch::stack({V_stay, V_up, V_down, V_left, V_right}, 1); // [B x 5 x W x H]

            auto Qsa = Rsa + gamma * Vs_next;

            auto Vnext = std::get<0>(torch::min(Qsa, 1));
            Vnext.index_put_({torch::arange(B), goal_grid_pos.index({torch::indexing::Slice(), 0}), 
                                                goal_grid_pos.index({torch::indexing::Slice(), 1})}, 0);

            auto err = torch::abs(V.slice(2, 1, -1).slice(3, 1, -1) - Vnext).max();
            if (err.item<double>() < tol)
                break;

            V.slice(2, 1, -1).slice(3, 1, -1) = Vnext;
        }
        return V.slice(2, 1, -1).slice(3, 1, -1);
    }

    std::unordered_map<std::string, torch::Tensor> get_key_metadata_map (const CostKeyDataHolder& data, 
                                                                const std::string& key) {
        return data.keys.at(key).metadata;
    }

    torch::Tensor get_key_data_tensor (const CostKeyDataHolder& data, 
                                const std::string& key) {
        return data.keys.at(key).data;
    }
};

#endif
    