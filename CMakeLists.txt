cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)

project(ComputeGraphBase)
add_compile_definitions(XTENSOR_USE_XSIMD=1)
file(GLOB_RECURSE SOURCE
        src/graph/*.cpp
        src/model/*.cpp
        src/util/*.cpp
        src/reinforcement/*.cpp
)
add_library(${PROJECT_NAME} ${SOURCE})
target_include_directories(${PROJECT_NAME} PUBLIC src)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC blas lapack)

target_include_directories(${PROJECT_NAME} PUBLIC extern/xsimd/include)
target_include_directories(${PROJECT_NAME} PUBLIC extern/xtl/include)
target_include_directories(${PROJECT_NAME} PUBLIC extern/xtensor/include)
target_include_directories(${PROJECT_NAME} PUBLIC extern/xtensor-blas/include)


project(ComputeGraphNN)
add_executable(${PROJECT_NAME} src/nnTest.cpp)
target_link_libraries(${PROJECT_NAME} ComputeGraphBase)

project(ComputeGraphQ)
add_executable(${PROJECT_NAME} src/qTest.cpp)
target_link_libraries(${PROJECT_NAME} ComputeGraphBase)


project(ComputeGraph)
