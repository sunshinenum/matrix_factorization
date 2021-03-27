cmake_minimum_required(VERSION 3.5)

project(MF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -g -pthread")

# SOURCE FILES
aux_source_directory(
        ${PROJECT_SOURCE_DIR}/src
        SRC
)

# HEADER FILES
include_directories(
        ${PROJECT_NAME}
        ${PROJECT_SOURCE_DIR}/include
)

# EXECUTABLE TARGET
add_executable(
        ${PROJECT_NAME}
        ${SRC}
)
