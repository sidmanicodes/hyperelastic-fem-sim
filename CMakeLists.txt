cmake_minimum_required(VERSION 3.16)
project(hyperelastic-fem-sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Consume Eigen as a library without configuring its tests and documentation.
set(BUILD_TESTING OFF CACHE BOOL "Build dependency tests")
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Build Eigen documentation")

include(FetchContent)

FetchContent_Declare(
    polyscope
    GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
    GIT_TAG master
)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG stable
)

FetchContent_MakeAvailable(polyscope)
FetchContent_MakeAvailable(eigen)
FetchContent_MakeAvailable(libigl)

add_executable(hyperelastic-fem-sim
    src/main.cpp
)

target_link_libraries(hyperelastic-fem-sim 
    PRIVATE 
        polyscope
        Eigen3::Eigen
        igl::core
)
