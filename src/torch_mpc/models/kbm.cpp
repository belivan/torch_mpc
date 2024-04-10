#include "base.h" // TODO: MAKE SURE I NEED THIS sadhfausdfhu
#include "kbm.h"
#include <torch/torch.h>
#include "yaml-cpp/yaml.h"
#include "yaml-cpp/node/parse.h"
#include "yaml-cpp/node/node.h"
#include "yaml-cpp/node/impl.h"
#include "yaml-cpp/node/node.h"
#include "yaml-cpp/parser.h"
#include <iostream>

int main()
{

	std::cout << "welcome to the program" << std::endl;
	//YAML::Node config = YAML::LoadFile("~/configs/costmap_speedmap.yaml");
	//printf("Loaded config\n");

	// figure out how to do the mpc setup thingy aaa, wait do i even need to do mpc setup ?

	std::cout << "config loaded yeah" << std::endl;

	//const std::string device_config = config["common"]["device"].as<std::string>();

	//const int B = config["common"]["B"].as<int>();
	//const int H = config["common"]["H"].as<int>();
	//const int M = config["common"]["M"].as<int>();
	//const double new_dt = config["common"]["dt"].as<double>();

	double L = 3.0;
	double min_throttle = 0.0;
	double max_throttle = 1.0;
	double max_steer = 0.3;
	double dt = 0.1;
	std::string device = "cpu";

	// Create an instance of KBM class
	KBM model(L, min_throttle, max_throttle, max_steer, dt, device);

	std::cout << "this is obviously making an instance" << std::endl;

	auto x0 = torch::zeros({ 3 });
	auto U = torch::zeros({ 50, 2 });

	U.index_put_({ torch::indexing::Slice(), 0 }, torch::ones({ 50 }));
	U.index_put_({ torch::indexing::Slice(), 1 }, 0.15 * torch::ones({ 50 }));

	std::cout << "it made the torch things correctly" << std::endl;

	auto t1 = std::chrono::high_resolution_clock::now();

	std::cout << "the time is crashing it! (cope)" << std::endl;

	auto X1 = model.rollout(x0, U);

	std::cout << "the model rollout is fine! (i dont believe you)" << std::endl;

	x0[-1] = -0.2; // i believe the pytorch api should support negative indexing in c++?

	std::cout << "you can do negative indexing!" << std::endl;
	/*
	if negative indexing doesn't work, use this code:
	int last_index = x0.size() - 1;
	x0[last_index] = -0.2;
	*/

	auto X2 = model.rollout(x0, U);

	x0[-1] = -0.2;
	auto X3 = model.rollout(x0, U);

	std::cout << "this is clearly a problem with saving" << std::endl;

	std::string dir_path = "C:\\Users\\13059";

	std::cout << "surely the string is not the problem" << std::endl;
	torch::save(X1, dir_path + "X1.pt");
	torch::save(X2, dir_path + "X2.pt");
	torch::save(X3, dir_path + "X3.pt");


	auto t2 = std::chrono::high_resolution_clock::now();
	//std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms to rollout KBM and save outputs \n" << std::endl;
	std::cout << "this is the end!" << std::endl;
	return 0;
}