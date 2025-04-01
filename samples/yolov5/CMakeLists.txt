cmake_minimum_required(VERSION 3.15.3)
project(yolov5)


# yolov5s
add_executable(yolov5s yolov5s.cc)
target_link_libraries(yolov5s ${OpenCV_LIBS} tcim_runtime_lite)
