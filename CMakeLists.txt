cmake_minimum_required(VERSION 3.10)
project(StereoNet_rknn)

set(CMAKE_CXX_STANDARD 14)

# opencv
find_package(OpenCV REQUIRED)

# npu
include_directories(/usr/include/)

add_executable(StereoNet_rknn main.cpp utils.cpp utils.h)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME}  /usr/lib/librknn_api.so)
target_link_libraries(${PROJECT_NAME}  /usr/lib/librknnrt.so)