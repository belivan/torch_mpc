#ifndef FOOTPRINT_SPEEDMAP_PROJECTION_IS_INCLUDED
#define FOOTPRINT_SPEEDMAP_PROJECTION_IS_INCLUDED

#include <torch/torch.h>
#include <cmath>
#include "base.h"
#include "utils.h"

class FootprintSpeedmapProjection : public CostTerm
{
    private:
        int speed_idx;
        double speed_margin;
        double sharpness;
        double length;
        double width;
        int nl;
        int nw;
        double length_offset;
        double width_offset;
        bool local_frame;
        std::vector<std::string> speedmap_key;
        torch::Device device;
        torch::Tensor footprint;

    public:
        // Args:
            // speed_idx: idx of state containing speed
            // speed_margin: actually command speedmap-this speed 
            // sharpness: sharpness of the barrier fn
            // length: the length of the footprint (along x)
            // width: the width of the footprint (along y)
            // nl: number of sample points along y
            // nw: number of sample points along x
            // length_offset: move the footprint length-wise by this amount
            // width_offset: move the footprint width-wise by this amount
            // local_frame: set this flag if the speedmap is in local frame
        FootprintSpeedmapProjection(int speed_idx=3, double speed_margin=1.0, double sharpness=5.0, 
                                    double length=5.0, double width=3.0, int nl=3, 
                                    int nw=3, double length_offset=-1.0, double width_offset=0.0, bool local_frame= false,
                                    const std::vector<std::string>& speedmap_key={"local_speedmap"}, 
                                    const torch::Device& device=torch::kCPU)
        : speed_margin(speed_margin), sharpness(sharpness), length(length), width(width),
        nl(nl), nw(nw), length_offset(length_offset), width_offset(width_offset),
        local_frame(local_frame), speedmap_key(speedmap_key), device(device) {
        footprint = make_footprint();}

        // ~FootprintSpeedmapProjection() = default;

        torch::Tensor make_footprint()
        {
            torch::Tensor x = torch::linspace(-length/2, length/2, nl, torch::TensorOptions().device(device)) + length_offset;
            torch::Tensor y = torch::linspace(-width/2, width/2, nw, torch::TensorOptions().device(device)) + width_offset;
            auto grid = torch::meshgrid({x, y});  
            torch::Tensor footprint = torch::stack({grid[0], grid[1]}, -1).view({-1,2});
            return footprint;
        }

        torch::Tensor apply_footprint(torch::Tensor traj)
        {
            /*
            Given a B x K x T x 3 tensor of states (last dim is [x, y, th]),
            return a B x K x T x F x 2 tensor of positions (F is each footprint sample)
            */
            torch::IntArrayRef tdims = traj.sizes().slice(0, traj.dim()-1);
            unsigned int nf = footprint.size(0);

            torch::Tensor pos = traj.slice(3, 0, 2);
            torch::Tensor th = traj.slice(3, 2, 3);

            torch::Tensor R = torch::stack({
                torch::stack({th.cos(), -th.sin()}, -1),
                torch::stack({th.sin(), th.cos()}, -1)
            }, -2); // [B x K x T x 2 x 2]

            auto R_expand = R.view({tdims[0], tdims[1], tdims[2], 1, 2, 2}); // [B x K x T x F x 2 x 2]
            auto footprint_expand = footprint.view({1, 1, 1, nf, 2, 1}); // [B x K x T x F x 2 x 1]

            auto footprint_rot = torch::matmul(R_expand, footprint_expand).view({tdims[0], tdims[1], tdims[2], nf, 2}); // [B x K x T x F x 2]
            auto footprint_traj = pos.view({tdims[0], tdims[1], tdims[2], 1, 2}) + footprint_rot; // [B x K x T x F x 2]

            return footprint_traj;
        }

        std::vector<std::string> get_data_keys() const override
        {
            return speedmap_key;
        }

        std::pair<torch::Tensor, torch::Tensor> cost(
            const torch::Tensor& states, 
            const torch::Tensor& actions, 
            const torch::Tensor& feasible, 
            const CostKeyDataHolder& data) override
        {
            torch::Tensor states2;
            if (local_frame)
            {
                states2 = utils::move_to_local_frame(states);
            }
            else
            {
                states2 = states;
            }

            torch::Tensor cost = torch::zeros({states2.size(0), states2.size(1)}, torch::TensorOptions().device(device));

            torch::Tensor speedmap = utils::get_data_tensor(data, speedmap_key[0]);
            std::unordered_map<std::string, torch::Tensor> metadata = utils::get_metadata_map(data, speedmap_key[0]);

            torch::Tensor world_pos = states2.index({"...", torch::indexing::Slice(), 
                                                            torch::indexing::Slice(0, 3)});
            torch::Tensor footprint_pos = apply_footprint(world_pos);
            auto results = utils::world_to_grid(footprint_pos, metadata);
            torch::Tensor grid_pos = std::get<0>(results);
            torch::Tensor invalid_mask = std::get<1>(results);

            // Switch grid axes to align with robot centric axes: +x forward, +y left
            grid_pos = grid_pos.index_select(-1, torch::tensor({1, 0}, torch::kLong));
            
            grid_pos.index_put_({invalid_mask}, 0);
            grid_pos = grid_pos.to(torch::kLong);

            auto idx0 = torch::arange(grid_pos.size(0), torch::TensorOptions().device(device));
            long ndims = grid_pos.dim() - 2;
            std::vector<int64_t> shape = {idx0.size(0)};
            for (int i = 0; i < ndims; ++i) {
                shape.push_back(1);
            }
            idx0 = idx0.view(shape);

            auto ref_speeds = torch::clone(speedmap.index({idx0, 
                                                    grid_pos.index({"...", torch::indexing::Slice(), 0}), 
                                                    grid_pos.index({"...", torch::indexing::Slice(), 1})}));

            ref_speeds.index_put_({invalid_mask}, 1e10);

            ref_speeds = std::get<0>(ref_speeds.min(-1));

            auto traj_speeds = states2.index({"...", speed_idx});

            torch::Tensor target_speeds = ref_speeds - speed_margin;
            cost = (sharpness * (traj_speeds - target_speeds)).exp().mean(-1);

            torch::Tensor state_feasible = traj_speeds < ref_speeds;

            torch::Tensor new_feasible = torch::logical_or(state_feasible.all(-1),
                                                    state_feasible.index({"...", 0}).logical_not());
            
            if (new_feasible.any().item().to<bool>() == false)
            {
                std::cout << "aaa" << std::endl;
            }

            return std::make_pair(cost, new_feasible);
        }

        FootprintSpeedmapProjection& to(const torch::Device& device) override
        {
            this->footprint = footprint.to(device);
            this->device = device;
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& os, const FootprintSpeedmapProjection& fsp);
};

std::ostream& operator<<(std::ostream& os, const FootprintSpeedmapProjection& fsp)
{
    os << "Footprint Speedmap Projection";
    return os;
}

#endif // FOOTPRINT_SPEEDMAP_PROJECTION_IS_INCLUDED
       