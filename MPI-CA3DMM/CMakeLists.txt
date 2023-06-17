cmake_minimum_required(VERSION 3.10)
SET(CMAKE_CXX_COMPILER mpicxx)   # use CC for okeanos

project(mpi_big_lab CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(.)

add_executable(ca3dmm
        densematgen.h
        densematgen.cpp
        main.cpp)
target_link_libraries(ca3dmm ${MPI_C_LIBRARIES})
