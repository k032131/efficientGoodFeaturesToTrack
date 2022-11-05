cmake_minimum_required(VERSION 2.8)

Project(eff)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS}) 

add_executable(eff efficientGoodFeaturesToTrack.cpp)

target_link_libraries(eff ${OpenCV_LIBS})
