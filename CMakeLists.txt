cmake_policy(SET CMP0054 NEW)
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(torch_mpc LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_subdirectory(../public/src ${CMAKE_BINARY_DIR}/public)

list(APPEND CMAKE_PREFIX_PATH "~/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG master # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_GetProperties(yaml-cpp)

if(NOT yaml-cpp_POPULATED)
  message(STATUS "Fetching yaml-cpp...")
  FetchContent_Populate(yaml-cpp)
  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

add_subdirectory(src/torch_mpc/algos)
# add_subdirectory(src/torch_mpc/models)
# add_subdirectory(src/torch_mpc/action_sampling)
# add_subdirectory(src/torch_mpc/cost_functions)