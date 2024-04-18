cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "/home/anton/Desktop/SPRING24/AEC/torch_mpc/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

list(APPEND CMAKE_PREFIX_PATH "/home/anton/Desktop/SPRING24/AEC/torch_mpc/yaml-cpp-master")
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

# add_subdirectory(sampling_strategies)
include_directories(sampling_strategies)

add_executable(generic_cost_function generic_cost_function.h generic_cost_function.cpp)
target_link_libraries(generic_cost_function ${TORCH_LIBRARIES} ${YAML_CPP_LIBRARIES})