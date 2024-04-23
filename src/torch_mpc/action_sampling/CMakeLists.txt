add_subdirectory(sampling_strategies)
include_directories(sampling_strategies)

add_executable(action_sampler action_sampler.h action_sampler.cpp)
target_link_libraries(action_sampler ${TORCH_LIBRARIES} ${YAML_CPP_LIBRARIES})