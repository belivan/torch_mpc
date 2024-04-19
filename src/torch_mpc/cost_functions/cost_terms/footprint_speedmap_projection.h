#ifndef FOOTPRINT_SPEEDMAP_PROJECTION_IS_INCLUDED
#define FOOTPRINT_SPEEDMAP_PROJECTION_IS_INCLUDED

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include "base.h"
#include <string>
#include <utility>

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
        std::vector<std::string> speedmap_key; // unsure if this should be a vector
        torch::Device device;
        torch::Tensor footprint;

    public:
        FootprintSpeedmapProjection(int speed_idx=3, double speed_margin=1.0, double sharpness=5.0, 
                                    double length=5.0, double width=3.0, int nl=3, 
                                    int nw=3, double length_offset=-1.0, double width_offset=0.0, bool local_frame= false,
                                    std::string speedmap_key="local", torch::Device device=torch::kCPU){
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

            this->speed_idx = speed_idx;
            this->speed_margin = speed_margin;
            this->sharpness = sharpness;
            this->length = length;
            this->width = width;
            this->nl = nl;
            this->nw = nw;
            this->length_offset = length_offset;
            this->width_offset = width_offset;
            this->local_frame = local_frame;
            this->speedmap_key = speedmap_key;
            this->device = device;
            footprint = make_footprint();
        }

        ~FootprintSpeedmapProjection() = default;

        torch::Tensor make_footprint()
        {
            torch::Tensor x = torch::linspace(-length/2, length/2, nl, torch::TensorOptions().device(device)) + length_offset;
            torch::Tensor y = torch::linspace(-width/2, width/2, nw, torch::TensorOptions().device(device)) + width_offset;
            torch::Tensor xx, yy;
            std::tie(xx, yy) = torch::meshgrid({x, y});  
            torch::Tensor footprint = torch::stack({xx, yy}, -1).view({-1,2});
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

            auto footprint_rot = torch::matmul(R_expand, footprint_exapnd).view({tdims[0], tdims[1], tdims[2], nf, 2}); // [B x K x T x F x 2]
            auto footprint_traj = pos.view({tdims[0], tdims[1], tdims[2], 1, 2}) + footprint_rot; // [B x K x T x F x 2]

            return footprint_traj;
        }

        std::vector<std::string> get_data_keys() override
        {
            return speedmap_key;
        }

        std::pair<torch::Tensor, torch::Tensor> cost(const torch::Tensor& states, const torch::Tensor& actions, 
                        const torch::Tensor& feasible, const std::unordered_map<std::string, torch::Tensor>& data) override
        {
            torch::Tensor states2;
            if (local_frame)
            {
                states2 = move_to_local_frame(states); // implement this function
            }
            else
            {
                states2 = states;
            }

            torch::Tensor cost = torch::zeros({states2.size(0), states2.size(1)}, torch::TensorOptions().device(device));

            // torch::Tensor speedmap = data.at(speedmap_key)["data"]; //Fix this
            // torch::Tensor metadata = data.at(speedmap_key)["metadata"];

            torch::Tensor world_pos = states2.index({"...", Slice(), Slice(None, 3)});
            torch::Tensor footprint_pos = apply_footprint(world_pos);
            auto results = world_to_grid(footprint_pos, metadata);
            torch::Tensor grid_pos = std::get<0>(results);
            torch::Tensor invalid_mask = std::get<1>(results);

            grid_pos = grid_pos.index({"...", Slice(), Slice(), {1,0}});
            
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
                                                            grid_pos.index({"...", Slice(), 0}), 
                                                            grid_pos.index({"...", Slice(), 1})}));
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

        FootprintSpeedmapProjectionn& to(torch::Device device) override
        {
            this->footprint = footprint.to(device);
            this->device = device;
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& os, const FootprintSpeedmapProjection& fsp)
        {
            os << "Footprint Speedmap Projection";
            return os;
        }
};

#endif
       