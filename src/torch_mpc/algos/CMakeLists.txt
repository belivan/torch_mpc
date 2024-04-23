include_directories(
        "../cost_functions" 
        "../cost_functions/cost_terms"
        "../models" "../" 
        "../sampling_strategies"
        "../sampling_strategies/sampling_strategies"
        )

add_executable(batch_sampling_mpc batch_sampling_mpc.cpp)
target_link_libraries(batch_sampling_mpc PUBLIC ${TORCH_LIBRARIES} yaml-cpp::yaml-cpp)