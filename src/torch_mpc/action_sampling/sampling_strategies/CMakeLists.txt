add_executable(uniform_gaussian uniform_gaussian.cpp uniform_gaussian.h)
target_link_libraries(uniform_gaussian "${TORCH_LIBRARIES}")

add_executable(gaussian_walk gaussian_walk.cpp gaussian_walk.h)
target_link_libraries(gaussian_walk "${TORCH_LIBRARIES}")

add_executable(action_library action_library.cpp action_library.h)
target_link_libraries(action_library "${TORCH_LIBRARIES}")


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET example-app
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:example-app>)
# endif (MSVC)