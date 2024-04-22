cmake_policy(SET CMP0054 NEW)
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(torch_mpc LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64")

# list(APPEND CMAKE_PREFIX_PATH "../../../../project_resources/libtorch-win-shared-with-deps-2.2.1+cu121/libtorch")
list(APPEND CMAKE_PREFIX_PATH "C:/Users/13059/Downloads/libtorch-win-shared-with-deps-2.2.2+cu121/libtorch")
find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# list(APPEND CMAKE_PREFIX_PATH "../../../../project_resources/yaml-cpp-master/yaml-cpp-master")
# find_package(yaml-cpp REQUIRED)
# include_directories(${YAML_CPP_INCLUDE_DIRS})

list(APPEND CMAKE_PREFIX_PATH "C:/Users/13059/anaconda3/Library/include/yaml-cpp")
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

# include(FetchContent)

# FetchContent_Declare(
#  yaml-cpp
#  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
#  GIT_TAG master # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
# )
# FetchContent_GetProperties(yaml-cpp)

# if(NOT yaml-cpp_POPULATED)
#  message(STATUS "Fetching yaml-cpp...")
#  FetchContent_Populate(yaml-cpp)
#  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
# endif()

# add_subdirectory(sampling_strategies)
include_directories(sampling_strategies)

add_executable(generic_cost_function generic_cost_function.h generic_cost_function.cpp)
# target_link_libraries(generic_cost_function PUBLIC ${TORCH_LIBRARIES} yaml-cpp::yaml-cpp)
target_link_libraries(generic_cost_function PUBLIC ${TORCH_LIBRARIES} ${YAML_CPP_LIBRARIES})

#The following code block is suggested to be used on Windows.
#According to https://github.com/pytorch/pytorch/issues/25457,
#the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET generic_cost_function
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:generic_cost_function>)
endif (MSVC)