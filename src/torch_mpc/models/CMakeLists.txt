cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(modelstesting)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "C:/Users/13059/torch_mpc/libtorch")
find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

list(APPEND CMAKE_PREFIX_PATH "C:/Users/13059/torch_mpc/yaml-cpp")
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

add_library(basemodel base.h base.cpp)

add_executable(kbmmodel kbm.h kbm.cpp)
target_link_libraries(kbmmodel basemodel ${TORCH_LIBRARIES} ${YAML_CPP_LIBRARIES})


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET kbmmodel
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:kbmmodel>)
endif (MSVC)