cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpp)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(cpp main.cpp)
target_link_libraries(cpp "${TORCH_LIBRARIES}")
set_property(TARGET cpp PROPERTY CXX_STANDARD 14)