cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)

project(ComputeGraph)

file(GLOB_RECURSE SOURCE src/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCE})
target_include_directories(${PROJECT_NAME} PUBLIC src)

find_package(Eigen3 REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIRS})