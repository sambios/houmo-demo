cmake_minimum_required(VERSION 3.15.3)
project(yolov8)

# yolov8
add_executable(yolov8m yolov8m.cc)
target_link_libraries(yolov8m ${OpenCV_LIBS} tcim_runtime_lite)