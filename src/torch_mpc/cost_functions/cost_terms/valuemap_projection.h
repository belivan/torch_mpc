#ifndef VALUEMAP_PROJECTION
#define VALUEMAP_PROJECTION

#include <torch/torch.h>
#include "base.h"

class ValuemapProjection
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
};



#endif