add_executable(euclidean_distance_to_goal euclidean_distance_to_goal.cpp)
target_link_libraries(euclidean_distance_to_goal "${TORCH_LIBRARIES}")

add_executable(euclidean_distance_to_multi_goal euclidean_distance_to_multi_goal.cpp)
target_link_libraries(euclidean_distance_to_multi_goal "${TORCH_LIBRARIES}")

add_executable(footprint_costmap_projection footprint_costmap_projection.cpp)
target_link_libraries(footprint_costmap_projection "${TORCH_LIBRARIES}")

add_executable(footprint_speedmap_projection footprint_speedmap_projection.cpp)
target_link_libraries(footprint_speedmap_projection "${TORCH_LIBRARIES}")

add_executable(goal_constraint goal_constraint.cpp)
target_link_libraries(goal_constraint "${TORCH_LIBRARIES}")

add_executable(speed_limit speed_limit.cpp)
target_link_libraries(speed_limit "${TORCH_LIBRARIES}")

add_executable(tip_over_cost tip_over_cost.cpp)
target_link_libraries(tip_over_cost "${TORCH_LIBRARIES}")

add_executable(unknown_map_projection unknown_map_projection.cpp)
target_link_libraries(unknown_map_projection "${TORCH_LIBRARIES}")

add_executable(valuemap_projection valuemap_projection.cpp)
target_link_libraries(valuemap_projection "${TORCH_LIBRARIES}")