#ifndef FOOTPRINT_COSTMAP_PROJECTION_IS_INCLUDED
#define FOOTPRINT_COSTMAP_PROJECTION_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <cmath>
#include "base.h"  // Ensure this file contains the definition of CostTerm
#include "utils.h" // Ensure this file contains necessary utility functions like world_to_grid
#include <iostream>
/*
    TODO: make sure to include--
     - torch_mpc.cost_functions.cost_terms.base [CostTerm]
     - torch_mpc.cost_functions.cost_terms.utils [world_to_grid, move_to_local_frame]

*/

using namespace torch::indexing;

class FootprintCostmapProjection : public CostTerm {
/*
    Stage cost that projects trajectories onto a costmap, but also applies a footprint
    */
private:
    float cost_thresh;
    float length;
    float width;
    int nl, nw;
    float length_offset;
    float width_offset;
    bool local_frame;
    std::vector<std::string> costmap_key;
    torch::Device device;
    torch::Tensor footprint;

public:
    /*
        Args:
            cost_thresh: any trajs that go through a cell w/ cost greater than this are marked infeasible
            length: the length of the footprint (along x)
            width: the width of the footprint (along y)
            nl: number of sample points along y
            nw: number of sample points along x
            length_offset: move the footprint length-wise by this amount
            width_offset: move the footprint width-wise by this amount
            local_frame: set this flag if costmap is in the local frame
            costmap_key: the key to look for costmap data in
        */
    FootprintCostmapProjection( float length = 5.0, float width = 3.0, 
                                float length_offset = -1.0, float width_offset = 0.0,
                                const torch::Device& device = torch::kCPU,
                                int nl = 3, int nw = 3,
                                float cost_thresh = 1e10,
                                bool local_frame = false,
                               const std::vector<std::string>& costmap_key = {"local_costmap"})
        : cost_thresh(cost_thresh), length(length), width(width), nl(nl), nw(nw),
          length_offset(length_offset), width_offset(width_offset), local_frame(local_frame),
          costmap_key(costmap_key), device(device) {
        footprint = make_footprint();
    }

    torch::Tensor make_footprint() {
        torch::Tensor xs = torch::linspace(-length / 2.0, length / 2.0, nl, torch::TensorOptions().device(device)) + length_offset;
        torch::Tensor ys = torch::linspace(-width / 2.0, width / 2.0, nw, torch::TensorOptions().device(device)) + width_offset;
        auto grid = torch::meshgrid({xs, ys});
        return torch::stack({grid[0], grid[1]}, -1).view({-1, 2});
    }

    torch::Tensor apply_footprint(const torch::Tensor& traj) {
        /*

        Given a B x K x T x 3 tensor of states (last dim is [x, y, th]),
        return a B x K x T x F x 2 tensor of positions (F is each footprint sample)

        */
        
        auto tdims = traj.sizes().vec();
        tdims.pop_back(); // remove the last dim

        int nf = footprint.size(0);

        auto pos = traj.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 2)});
        auto th = traj.index({torch::indexing::Ellipsis, 2});

        auto R = torch::stack({
            torch::stack({th.cos(), -th.sin()}, -1),
            torch::stack({th.sin(), th.cos()}, -1)
        }, -2); // [B x K x T x 2 x 2]
        std::vector<int64_t> new_shape = {tdims[0], tdims[1], tdims[2], 1, 2, 2};
        auto R_expand = R.view(new_shape);
        auto footprint_expand = footprint.view({1, 1, 1, nf, 2, 1});

        if(R_expand.dtype() != footprint_expand.dtype())
        {
            R_expand = R_expand.to(footprint_expand.dtype());
        }
        auto footprint_rot = torch::matmul(R_expand, footprint_expand).view({tdims[0], tdims[1], tdims[2], nf, 2}); // expected SCALAR TYPE DOUBLE BUT FOUND FLOAT / new error expected Float but found Double
        auto footprint_traj = pos.view({tdims[0], tdims[1], tdims[2], 1, 2}) + footprint_rot;

        return footprint_traj;
    }

    std::vector<std::string> get_data_keys() const override {
        return costmap_key;
    }

    std::pair<torch::Tensor, torch::Tensor> cost(const torch::Tensor& states, const torch::Tensor& actions,
                                                 const torch::Tensor& feasible, const CostKeyDataHolder& data) override {
       
        torch::Tensor states2 = local_frame ? utils::move_to_local_frame(states) : states;

        torch::Tensor cost = torch::zeros({states2.size(0), states2.size(1)}, torch::TensorOptions().device(device));
        torch::Tensor costmap = utils::get_key_data_tensor(data, costmap_key[0]);
        std::unordered_map<std::string, torch::Tensor> metadata = utils::get_key_metadata_map(data, costmap_key[0]);

        // get world_pos
        torch::Tensor world_pos = states2.index({"...", torch::indexing::Slice(0, 3)});
        // get the footprint
        torch::Tensor footprint_pos = apply_footprint(world_pos); // IS IT HERE? EXPECTED SCALAR TYPE DOUBLE BUT FOUND FLOAT
        // footprint -> grid positions
        auto [grid_pos, invalid_mask] = utils::world_to_grid(footprint_pos, metadata);
        // roboaxes
        grid_pos = grid_pos.index_select(-1, torch::tensor({1, 0}, torch::kLong).to(device));
        // uhh invalid costmap
        // std::cout << "grid_pos: " << grid_pos.sizes() << std::endl;
        // std::cout << "invalid_mask: " << invalid_mask.sizes() << std::endl;
        grid_pos.masked_fill_(invalid_mask.unsqueeze(-1), 0);
        grid_pos = grid_pos.to(torch::kLong);

        torch::Tensor idx0 = torch::arange(grid_pos.size(0), torch::TensorOptions().device(device));
        int ndims = grid_pos.dim() - 2;
        std::vector<int64_t> shape(1, idx0.size(0));
        shape.insert(shape.end(), ndims, 1);
        idx0 = idx0.view(shape);

        // std::cout << "idx0: " << idx0.sizes() << std::endl;
        // std::cout << "grid_pos: " << grid_pos.sizes() << std::endl;
        // std::cout << "costmap: " << costmap.sizes() << std::endl;

        auto new_costs = costmap.index({idx0, grid_pos.index({"...", 0}), grid_pos.index({"...", 1})}).clone();
        // std::cout << "new_costs: " << new_costs.sizes() << std::endl;
        new_costs.masked_fill_(invalid_mask, 0.0);

        // thresholding
        auto new_feasible = new_costs.lt(cost_thresh).all(-1).all(-1); //IT CRASHES HERE, why
        // sum over time
        cost += new_costs.mean(-1).sum(-1);
        // std::cout << "Return Costmap" << std::endl;
        return {cost, new_feasible};
    }

    FootprintCostmapProjection& to(const torch::Device& device) override {
        this->device = device;
        this->footprint = this->footprint.to(device);
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const FootprintCostmapProjection& fcp) {
        os << "Footprint Costmap Projection";
        return os;
    }
};

#endif // FOOTPRINT_COSTMAP_PROJECTION_IS_INCLUDED