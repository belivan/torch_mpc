#include "setup_mpc.h"

int main() {
    std::string config_file = "config.yaml"; // specify the path to the config file

    YAML::Node config = YAML::LoadFile(config_file);
    BatchSamplingMPC mpc = setup_mpc(config);

    std::cout << mpc << std::endl;

    return 0;
}