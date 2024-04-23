include_directories(
        "../cost_functions" 
        "../cost_functions/cost_terms"
        "../models" "../" 
        "../sampling_strategies"
        "../sampling_strategies/sampling_strategies"
        )

add_executable(batch_sampling_mpc batch_sampling_mpc.cpp)
target_link_libraries(batch_sampling_mpc PUBLIC ${TORCH_LIBRARIES} yaml-cpp::yaml-cpp)

add_test(NAME batch_sampling_mpc COMMAND batch_sampling_mpc)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET batch_sampling_mpc
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR: batch_sampling_mpc>)
endif (MSVC)

enable_testing()