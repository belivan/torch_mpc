#include <torch/torch.h>
#include <cmath>
#include "base.h"

namespace indexing = torch::indexing;

class GravityThrottleKBM : public Model
{
private:
    double L; // unsure if should be float or double here; float may be faster
    double dt;
    std::string state_key;
    std::string pitch_key;
    std::string steer_key;
    std::string device;

    std::vector<double> u_ub;
    std::vector<double>u_lb; // unsure if this makes sense

    double steer_rate_lim;
    std::vector<double> actuator_model;
    bool requires_grad;
    int num_params;

// TODO: make sure to add const/override/both whereever necessary
public:
    GravityThrottleKBM(const std::vector<double>& actuator_model, 
                        double L=3.0, std::vector<double> throttle_lim={0., 1.0}, 
                        std::vector<double> steer_lim={-0.52, 0.52},
                        double steer_rate_lim=0.45, 
                        double dt=0.1, std::string device='cpu',
                        bool requires_grad=false, 
                        std::string state_key="state", 
                        std::string pitch_key="pitch", 
                        std::string steer_key="steer_angle") :
        L(L),
        dt(dt),
        state_key(state_key),
        pitch_key(pitch_key),
        steer_key(steer_key),
        steer_rate_lim(steer_rate_lim),
        requires_grad(requires_grad),
        num_params(actuator_model.size())
    {

        this->actuator_model = actuator_model;
        std::vector<double> u_lb = {throttle_lim[0], steer_lim[0]};
        std::vector<double> u_ub = {throttle_lim[1], steer_lim[1]};

        std::vector<double> x_lb(5, -std::numeric_limits<double>::infinity());
        std::vector<double> x_ub(5, std::numeric_limits<double>::infinity());
    }


    torch::Tensor dynamics(torch::Tensor state, torch::Tensor action)
    {
        auto statepermute = 
    }

    torch::Tensor predict(torch::Tensor state, torch::Tensor action)
    {

    }

    torch::Tensor rollout(torch::Tensor state, torch::Tensor actions)
}