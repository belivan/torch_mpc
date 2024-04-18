#ifndef UNKNOWN_MAP_PROJECTION_IS_INCLUDED
#define UNKNOWN_MAP_PROJECTION_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "base.h"
#include <string>
#include <utility>

class UnknownMapProjection
{
    private:
    int start_idx;
    double unknown_threshold;
    double unknown_penalty;
    torch::Device device;

public:
    UnknownMapProjection(int start_idx = 0, double unknown_threshold = 0.05, double unknown_penalty = 1e8, const torch::Device device = torch::kCPU)
    : start_idx(start_idx), unknown_threshold(unknown_threshold), unknown_penalty(unknown_penalty), device(device) {}

    std::vector<std::string> get_data_keys() const {
        return {"unknown_map", "unknown_map_metadata"};
    }

    torch::Tensor cost(const torch::Tensor& states, const torch::Tensor& actions, 
                    const std::unordered_map<std::string, torch::Tensor>& data) {
        torch::Tensor cost = torch::zeros({states.size(0), states.size(1)}, device);
        
        // see if this works
        torch::Tensor costmap = data.at("unknown_map");
        torch::Tensor metadata = data.at("unknown_map_metadata");

        torch::Tensor world_pos = states.index({"...", torch::indexing::Slice(None, 2)});
        auto grid_result = world_to_grid(world_pos, metadata);  // Assume world_to_grid is implemented
        torch::Tensor grid_pos = std::get<0>(grid_result);
        torch::Tensor invalid_mask = std::get<1>(grid_result);

        // Switch grid axes to align with robot centric axes: +x forward, +y left
        grid_pos = grid_pos.index({"...", {1, 0}});

        // Handle invalid indices
        grid_pos.index_put_({invalid_mask}, 0);
        grid_pos = grid_pos.to(torch::kLong);

        torch::Tensor idx0 = torch::arange(grid_pos.size(0), device);
        std::vector<int64_t> shape = {idx0.size(0)};
        for (int i = 0; i < grid_pos.dim() - 2; ++i) {
            shape.push_back(1);
        }
        idx0 = idx0.view(shape);

        torch::Tensor unknowns = torch::clone(costmap.index({idx0, grid_pos.index({"...", 0}), grid_pos.index({"...", 1})}));
        unknowns.index_put_({invalid_mask}, 1);

        torch::Tensor is_unknown_mask = (unknowns.index({torch::indexing::Slice(), torch::indexing::Slice(), 
                                    torch::indexing::Slice(start_idx, torch::indexing::None)}).mean(-1) > unknown_threshold);

        cost.index_put_({is_unknown_mask}, unknown_penalty);

        return cost;
    }

    UnknownMapProjection& (const torch::Device& device) {
        this->device = device;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const UnknownMapProjection& ump) {
        os << "Unknown Map Projection";
        return os;
    }
};

#endif