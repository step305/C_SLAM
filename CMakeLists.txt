cmake_minimum_required(VERSION 3.10)
project(C_SLAM)

set(CMAKE_CXX_STANDARD 14)
add_executable(C_SLAM main.cpp)

find_package(OpenCV REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(C_SLAM ${OpenCV_LIBS})

