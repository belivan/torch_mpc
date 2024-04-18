#ifndef UTILS_IS_INCLUDED
#define UTILS_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <utility>

namespace utils
{
    torch::Tensor move_to_local_frame(const torch::Tensor& traj, int xidx = 0, int yidx = 1, int thidx = 2) {
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
    auto ops = traj.index({"...", 0, {xidx, yidx}}).view({traj.size(0), traj.size(1), 1, 2});
    auto oths = traj.index({"...", 0, thidx}).view({traj.size(0), traj.size(1), 1, 1});

    // translate
    auto pos_out = traj.index({"...", {xidx, yidx}}) - ops;

    // rotate
    auto th_out = traj.index({"...", thidx}) - oths;

    auto cos_th = torch::cos(oths);
    auto sin_th = torch::sin(oths);
    auto R = torch::stack({torch::stack({cos_th, -sin_th}, -1),
                           torch::stack({sin_th, cos_th}, -1)}, -1);

    auto pos_out = torch::matmul(R, positions.view({-1, 2, 1})).view_as(pos_out); // [B x K x T x 2] check this

    auto traj_out = traj.clone();
    traj_out.index_put_({"...", xidx}, pos_out.index({"...", 0}));
    traj_out.index_put_({"...", yidx}, pos_out.index({"...", 1}));
    traj_out.index_put_({"...", thidx}, th_out.index({"...", 0}));
    return traj_out;
    }

    std::pair<torch::Tensor, torch::Tensor> world_to_grid(const torch::Tensor& world_pos, const torch::Tensor& metadata) {
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
    auto res = metadata["resolution"];
    auto nx = (metadata["length_x"] / res).to(torch::kLong);
    auto ny = (metadata["length_y"] / res).to(torch::kLong);
    auto ox = metadata["origin"].index({0});
    auto oy = metadata["origin"].index({1});

    auto gx = (world_pos.index({"...", 0}) - ox.unsqueeze(-1)) / res;
    auto gy = (world_pos.index({"...", 1}) - oy.unsqueeze(-1)) / res;

    auto grid_pos = torch::stack({gx, gy}, -1).to(torch::kLong);
    auto invalid_mask = (grid_pos.index({"...", 0}) < 0) |
                        (grid_pos.index({"...", 1}) < 0) |
                        (grid_pos.index({"...", 0}) >= nx.unsqueeze(-1)) |
                        (grid_pos.index({"...", 1}) >= ny.unsqueeze(-1));

    return std::make_pair(grid_pos, invalid_mask);
    }   

    torch::Tensor value_iteration(const torch::Tensor& costmap, 
                            const torch::Tensor& metadata, 
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
    auto res = metadata["resolution"];
    auto nx = (metadata["height"] / res).to(torch::kLong)[0];
    auto ny = (metadata["width"] / res).to(torch::kLong)[0];
    
    // setup
    auto V = torch::full({B, nx+2, ny+2}, 1e10, costmap.options());
    auto R = torch::full({B, nx+2, ny+2}, 1e10, costmap.options());
    R.slice(1, 1, -1).slice(2, 1, -1) = costmap;

    // load in goal point
    auto goal_grid_pos = world_to_grid(goals.unsqueeze(1).unsqueeze(1), metadata).first.squeeze(1);
    R.index_put_({torch::arange(B), goal_grid_pos.index({Slice(), 0}) + 1, goal_grid_pos.index({Slice(), 1}) + 1}, 0);

    // perform value iteration
    for (int i = 0; i < nx + ny; ++i) {
        auto Rsa = torch::stack({costmap,
                                costmap,
                                costmap,
                                costmap,
                                costmap}, 1);

        // handle terminal state
        V.index_put_({torch::arange(B), goal_grid_pos.index({Slice(), 0}+1), goal_grid_pos.index({Slice(), 1})+1}, 0);

        auto V_stay = V.slice(2, 1, -1).slice(3, 1, -1);
        auto V_up = V.slice(2, 2, torch::indexing::None).slice(3, 1, -1);
        auto V_down = V.slice(2, 0, -2).slice(3, 1, -1);
        auto V_left = V.slice(2, 1, -1).slice(3, 2, torch::indexing::None);
        auto V_right = V.slice(2, 1, -1).slice(3, 0, -2);

        auto Vs_next = torch::stack({V_stay, V_up, V_down, V_left, V_right}, 1); // [B x 5 x W x H]

        auto Qsa = Rsa + gamma * Vs_next;
        // auto Qsa = R.slice(1, 1, -1).slice(2, 1, -1).unsqueeze(1) + gamma * Vs_next;

        auto Vnext = std::get<0>(torch::min(Qsa, 1));
        Vnext.index_put_({torch::arange(B), goal_grid_pos.index({Slice(), 0}), 
                                            goal_grid_pos.index({Slice(), 1})}, 0);

        auto err = torch::abs(V.slice(2, 1, -1).slice(3, 1, -1) - Vnext).max();
        if (err.item<double>() < tol)
            break;

        V.slice(2, 1, -1).slice(3, 1, -1) = Vnext;
    }
    return V.slice(2, 1, -1).slice(3, 1, -1);
    }
};

#endif
    