#include "base.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>

class UniformGaussion: public SamplingStrategy
{
    /*
    Sampling strategy that applies adds gaussian noise to the nominal sequence
    */
    private:
        // int B;
        // int K;
        // int H;
        // int M;
        // torch::Device device;
    public:
        UniformGaussion(const std::vector<float> scale, cosnt int B, const int K, const int H, const int M, const torch::Device device)
        : SamplingStrategy(B, K, H, M, device),
        scale(setup_scale(scale)){}  // check if this is the right way to do this

        torch::Tensor setup_scale(const std::vector<float> scale)
        {
            auto scale_holder = torch::tensor(scale, torch::dtype(torch::kFloat32).device(device));
            // assert len(scale) == self.M, "scale has dimension {}. Expected {}".format(actlib.shape[2], self.M)
            assert(scale_holder.size() == M, "scale has dimension {}. Expected {}".format(scale_holder.size(), M));
            assert(torch::all(scale_holder >= 0).item<bool>(), "got negative scale");
            // is this the right way to do this?

            return scale_holder;
        }

        torch::Tensor sample(const torch::Tensor &u_nominal, const torch::Tensor &u_lb, const torch::Tensor &u_ub)
        {
            auto noise = torch::randn({B, K, H, M}, torch::dtype(torch::kFloat32).device(device));
            auto noise *= scale.view({1,1,1,M});

            auto samples = u_nominal.view({B, 1, H, M}) + noise;
            // auto samples_clip = clip_samples(samples, u_lb, u_ub); // figure out how to implement this
            return samples_clip;
        }

        UniformGaussion& to(const torch::Device &device) override
        {
            this->device = device;
            this->scale = this->scale.to(device);
            return *this;
        }
};
    
int main()
{
    std::vector<float> scale = {0.1, 0.5};

    int B = 3;
    int K = 23;
    int H = 54;
    int M = 2;

    auto sampling_strategy = UniformGaussion(scale, B, K, H, M, torch::GPU(torch::kCUDA, 0));

    auto nom = torch::zeros({B, H, M}, torch::dtype(torch::kFloat32).device(sampling_strategy.device));
    auto ulb = torch::ones({B, M}, torch::dtype(torch::kFloat32).device(sampling_strategy.device));
    auto uub = torch::ones({B, M}, torch::dtype(torch::kFloat32).device(sampling_strategy.device));

    ulb[0] *= 0.1;
    uub[0] *= 0.1;

    auto samples = sampling_strategy.sample(nom, ulb, uub);

    std::cout << samples.sizes() << std::endl;

    auto sample_min = std::get<0>(samples.min(1)).min(1)[0];
    auto sample_max = std::get<0>(samples.max(1)).max(1)[0];

    std::cout << sample_min << std::endl;
    std::cout << sample_max << std::endl;

    return 0;
}

        