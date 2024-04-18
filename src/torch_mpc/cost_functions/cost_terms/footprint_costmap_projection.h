#ifndef FOOTPRINT_COSTMAP_PROJECTION_IS_INCLUDED
#define FOOTPRINT_COSTMAP_PROJECTION_IS_INCLUDED

#include <torch/torch.h>
#include <string>
#include <utils.h> // THIS IS ASSUMING THAT WE ARE GOING TO MAKE A UTILS.H AAAAA

/*
    TODO: make sure to include--
     - torch_mpc.cost_functions.cost_terms.base [CostTerm]
     - torch_mpc.cost_functions.cost_terms.utils [world_to_grid, move_to_local_frame]

*/
using namespace torch::indexing;

class FootprintCostmapProjection : public CostTerm
{
    /*
    Stage cost that projects trajectories onto a costmap, but also applies a footprint
    */
private:
    float cost_thresh, length, width, length_offset, width_offset;
    int nl, nw; 
    bool local_frame;
    std::vector<std::string> costmap_key;
    torch::Device device;
    torch::Tensor footprint;

public:

    FootprintCostmapProjection(float cost_thresh = 1e10, float length = 5.0, float width = 3.0, int nl = 3, int nw = 3, 
                            float length_offset = -1.0, float width_offset = 0.0, bool local_frame = false, 
                            std::vector<std::string> costmap_key = "local_costmap", torch::Device device = "cpu")
    : cost_thresh(cost_thresh), length(length), width(width), nl(nl), nw(nw), length_offset(length_offset), 
        width_offset(width_offset), local_frame(local_frame), costmap_key(costmap_key), 
        device(torch::Device(device))
    {
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

        footprint = make_footprint();
    }

    torch::Tensor make_footprint(void)
    {
        torch::Tensor xs = torch::linspace({-length / 2.0, length / 2.0, nl}, torch::TensorOptions().device(device)) + length_offset;
        torch::Tensor ys = torch::linspace({-width / 2.0, width / 2.0, nw}, torch::TensorOptions().device(device)) + width_offset;

       return torch::stack(torch::meshgrid(xs, ys)).view({-1, 2}); // unsure if i can set indexing here?
    }

    torch::Tensor apply_footprint(torch::Tensor traj)
    {
        /*

        Given a B x K x T x 3 tensor of states (last dim is [x, y, th]),
        return a B x K x T x F x 2 tensor of positions (F is each footprint sample)

        */

        torch::IntArrayRef tdims = traj.sizes().slice(0, -1);
        int nf = footprint.size(0);
        auto pos = traj.index({torch::indexing::Ellipsis, torch::indexing::Slice(torch::None, 2)});
        // torch::Tensor pos = traj.index({"...", 1, 2}); // unsure how to get the first 2
        // torch::Tensor pos = traj.index("...", Slice(torch::None, 2)); // TODO: CHECK THE WAY I SLICED THIS WHYYYYYY
        torch::Tensor th = traj.index("...", 2);

        torch::Tensor R = torch::stack({torch::stack({th.cos(), -th.sin()}, -1), 
                                    torch::stack({th.sin(), th.cos()}, -1)}, -2);

        torch::Tensor R_expand = R.view(tdims + torch::IntArrayRef({1, 2, 2}));
        torch::Tensor footprint_expand = footprint.view(torch::IntArrayRef({1, 1, 1, nf, 2, 1}));

        torch::Tensor footprint_rot = torch::matmul(R_expand, footprint_expand).view(tdims + torch::IntArrayRef({nf, 2}));
        torch::Tensor footprint_traj = pos.view(tdims + torch::IntArrayRef({1, 2})) + footprint_rot;

        return footprint_traj;
    }

    std::vector<std::string> get_data_keys() override
    {
        return costmap_key;
    }

    // // TODO: double check that it does not need to be a std::pair<torch::Tensor, torch::Tensor>
    // std::pair cost(torch::Tensor states, torch::Tensor actions, torch::Tensor feasible, torch::Tensor data) override
    // {
    //     torch::Tensor cost = torch.zeros({states.size(0), states.size(1)}, torch::TensorOptions().device(device));

    //     // auto costmap = data.index(costmap_key, 'data')
    //     torch::Tensor costmap = data[costmap_key + "_data"];
    //     torch::Tensor metadata = data[costmap_key + "_metadata"];


    //     return std::make_pair(cost, new_feasible)
    // }
    std::pair<torch::Tensor, torch::Tensor> cost(torch::Tensor states, torch::Tensor actions, torch::Tensor feasible, torch::Tensor data) override
    {
        // move to local frame if necessary
        torch::Tensor states2 = local_frame ? move_to_local_frame(states) : states;

        // zeros init
        torch::Tensor cost = torch::zeros({states2.size(0), states2.size(1)}, torch::TensorOptions().device(device));

        // get costmap/metadata from data
        torch::Tensor costmap = data[costmap_key + "_data"];
        torch::Tensor metadata = data[costmap_key + "_metadata"];

        // get world_pos
        torch::Tensor world_pos = states2.index({torch::indexing::Ellipsis, torch::indexing::Slice(torch::None, 3)}); // Assuming the last dimension contains x, y, and th

        // get the footprint
        torch::Tensor footprint_pos = apply_footprint(world_pos); // Assuming you have implemented apply_footprint function

        // footprint -> grid positions
        torch::Tensor grid_pos, invalid_mask;
        std::tie(grid_pos, invalid_mask) = world_to_grid(footprint_pos, metadata); // Assuming you have implemented world_to_grid function

        // roboaxes
        grid_pos = grid_pos.index({torch::indexing::Ellipsis, {1, 0}});

        // uhh invalid costmap
        grid_pos.masked_fill_(invalid_mask, 0);
        grid_pos = grid_pos.to(torch::kLong);
        torch::Tensor idx0 = torch::arange(grid_pos.size(0)).view({grid_pos.size(0), 1});
        idx0 = idx0.unsqueeze(-1).expand({idx0.size(0), grid_pos.size(1), grid_pos.size(2)});
        
        torch::Tensor new_costs = costmap.index({idx0, grid_pos.index({torch::indexing::Slice(None), 0}), grid_pos.index({torch::indexing::Slice(None), 1})}).clone();
        new_costs.masked_fill_(invalid_mask, 0.);

        // thresholding
        torch::Tensor new_feasible = new_costs.lt(cost_thresh).all({-1, -2});

        // sum over time
        cost += new_costs.mean({-1}).sum({-1});

        return std::make_pair(cost, new_feasible);
    }

    FootprintCostmapProjection& to(torch::Device device) override
    {
        this->device = device;
        return *this;
    }

    std::string __repr__() const override
    {
        return "Costmap Projection";
    }

;
}

#endif