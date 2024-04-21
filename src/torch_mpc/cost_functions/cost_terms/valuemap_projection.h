#ifndef VALUEMAP_PROJECTION
#define VALUEMAP_PROJECTION

#include <torch/torch.h>
#include "base.h"
#include <string>
#include <unordered_map>

using namespace torch::indexing;

class ValuemapProjection : public CostTerm
{
private:
    float length, width, length_offset, width_offset;
    int nl, nw;
    bool local_frame, final_state;
    std::string valuemap_key;
    torch::Device device;
    torch::Tensor footprint;
public:
    // this code is more or less similar to tipovercost, i would just copy from there when someone has the time
    ValuemapProjection(float length, float width, int nl, int nw,
        float length_offset, float width_offset,
        bool local_frame, std::string valuemap_key,
        bool final_state, torch::Device device)
        : length(length), width(width), length_offset(length_offset), width_offset(width_offset),
        nl(nl), nw(nw), local_frame(local_frame), valuemap_key(valuemap_key),
        final_state(final_state), device(device)
    {
        footprint = make_footprint();
    }

    torch::Tensor make_footprint(void) const
    {
        auto xs = torch::linspace(-length / 2., length / 2., nl, torch::TensorOptions(device)) + length_offset;
        auto ys = torch::linspace(-width / 2., width / 2., nw, torch::TensorOptions(device)) + width_offset;

        // i believe libtorch does not have a meshgrid
        auto grid_xs = xs.unsqueeze(1).expand({ -1, ys.size(0) });
        auto grid_ys = ys.unsqueeze(0).expand({ xs.size(0), -1 });

        auto footprint = torch::stack({ grid_xs, grid_ys }, -1).view({ -1, 2 });

        return footprint;
    }

    torch::Tensor apply_footprint(torch::Tensor traj) {
        auto tdims = traj.sizes().slice(0, 0, traj.dim() - 1);
        auto nf = footprint.size(0);

        auto pos = traj.index({ Ellipsis, Slice(None, 2) });
        auto th = traj.index({ Ellipsis, 2 });

        // rotation matrix R
        auto R = torch::stack({ th.cos(), -th.sin(), th.sin(), th.cos() }, -1).view({ -1, 2, 2 });

        auto R_expand = R.view({ tdims.size(0), tdims.size(1), 1, 2, 2 });
        auto footprint_expand = footprint.view({ 1, 1, 1, nf, 2, 1 });

        auto footprint_rot = torch::matmul(R_expand, footprint_expand).view({ tdims.size(0), tdims.size(1), nf, 2 });
        auto footprint_traj = pos.view({ tdims.size(0), tdims.size(1), 1, 2 }) + footprint_rot;

        return footprint_traj;
    }

    torch::Tensor cost(const torch::Tensor& states, const torch::Tensor& actions, const std::unordered_map<std::string, torch::Tensor>& data, torch::Tensor past_cost = torch::Tensor(), torch::Tensor cur_obs = torch::Tensor()) 
    {
        torch::Device device = states.device();
        torch::Tensor cost = torch::zeros({ states.size(0), states.size(1) }, torch::TensorOptions().device(device));

        torch::Tensor world_speed = states.index({ Ellipsis, 3 }).abs();
        torch::Tensor future_steer = states.index({ Ellipsis, 4 });

        torch::Tensor cur_steer;
        if (cur_obs.is_dict()) {
            cur_steer = cur_obs["new_state"].index({ Ellipsis, -1 });
        }
        else {
            cur_steer = cur_obs.index({ Ellipsis, -1 });
        }

        torch::Tensor init_change_in_steer = actions.index({ Ellipsis, 0, -1 }) - cur_steer;

        torch::Tensor max_del_steer = torch::zeros_like(init_change_in_steer);
        torch::Tensor max_steer = torch::zeros_like(actions.index({ Ellipsis, -1 }));
        for (int i = 0; i < self.max_del_steer_vel_range.size(); ++i) {
            torch::Tensor mask;
            if (i == 0) {
                mask = world_speed < self.max_del_steer_vel_range[i];
            }
            else {
                mask = world_speed < self.max_del_steer_vel_range[i] &&
                    world_speed > self.max_del_steer_vel_range[i - 1];
            }
            max_del_steer.masked_fill_(mask.index({ Ellipsis, 0 }), self.max_del_steer[i]);
            max_steer.masked_fill_(mask, self.max_steer[i]);
        }
        torch::Tensor remaining_mask = max_del_steer == 0;
        max_del_steer.masked_fill_(remaining_mask, self.max_del_steer[-1]);
        max_steer.masked_fill_(remaining_mask, self.max_steer[-1]);

        cost += self.alpha_del_steer * (init_change_in_steer.abs() > max_del_steer);

        if (self.max_yaw_rate > 0) {
            torch::Tensor yaw_rate = world_speed * torch::tan(future_steer) / self.L;
            cost += self.alpha_yaw_rate * (yaw_rate.abs() > self.max_yaw_rate).sum(-1);
        }

        cost += self.alpha_del_steer * (states.index({ Ellipsis, -1 }).abs() > max_steer).sum(-1);

        return cost;
    }


    ValuemapProjection to(torch::Device device)
    {
        footprint = footprint.to(device);
        this->device = device;
        return *this;
    }

    // overloaded printing operator
    friend std::ostream& operator<<(std::ostream& os, const ValuemapProjection& vp);
};

std::ostream& operator<<(std::ostream& os, const ValuemapProjection& vp)
{
    //os << "ValuemapProjection: length=" << vp.length << ", width=" << vp.width << ", ...";
    //return os;
    os << "Costmap Projection";
    return 
}



#endif