cmake_policy(SET CMP0054 NEW)
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(torch_mpc LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "../../../../project_resources/libtorch-win-shared-with-deps-2.2.1+cu121/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# list(APPEND CMAKE_PREFIX_PATH "../../../../project_resources/yaml-cpp-master/yaml-cpp-master")
# find_package(yaml-cpp REQUIRED)
# include_directories(${YAML_CPP_INCLUDE_DIRS})
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

include_directories(
        "../cost_functions" 
        "../cost_functions/cost_terms"
        "../models" "../" 
        "../sampling_strategies"
        "../sampling_strategies/sampling_strategies"
        )

add_executable(batch_sampling_mpc batch_sampling_mpc.cpp)
target_link_libraries(batch_sampling_mpc PUBLIC ${TORCH_LIBRARIES} yaml-cpp::yaml-cpp)

#The following code block is suggested to be used on Windows.
#According to https://github.com/pytorch/pytorch/issues/25457,
#the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET batch_sampling_mpc
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:batch_sampling_mpc>)
endif (MSVC)