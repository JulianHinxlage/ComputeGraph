cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)

project(ComputeGraph)

file(GLOB_RECURSE SOURCE src/*.cpp)
file(GLOB_RECURSE XSOURCE extern/xtensor-blas/include/*.cxx)
add_executable(${PROJECT_NAME} ${SOURCE} ${XSOURCE})
target_include_directories(${PROJECT_NAME} PUBLIC src)

find_package(BLAS REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC blas)

add_subdirectory(extern/xsimd)
add_subdirectory(extern/xtl)
add_subdirectory(extern/xtensor)
target_include_directories(${PROJECT_NAME} PUBLIC extern/xsimd/include)
target_include_directories(${PROJECT_NAME} PUBLIC extern/xtl/include)
target_include_directories(${PROJECT_NAME} PUBLIC extern/xtensor/include)
target_include_directories(${PROJECT_NAME} PUBLIC extern/xtensor-blas/include)
