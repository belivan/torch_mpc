add_executable(uniform_gaussian uniform_gaussian.cpp)
target_link_libraries(uniform_gaussian "${TORCH_LIBRARIES}")

add_executable(gaussian_walk gaussian_walk.cpp)
target_link_libraries(gaussian_walk "${TORCH_LIBRARIES}")

add_executable(action_library action_library.cpp)
target_link_libraries(action_library "${TORCH_LIBRARIES}")