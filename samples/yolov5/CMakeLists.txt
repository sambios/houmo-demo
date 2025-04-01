cmake_minimum_required(VERSION 3.15.3)
project(yolo)



# yolov5
add_executable(yolov5 yolov5s.cc)
target_link_libraries(yolov5 ${OpenCV_LIBS} ${TCIM_LIBRARY})



#add_executable(yolov8_onnx yolov8_onnx.cpp)
#target_link_libraries(yolov8_onnx onnxruntime ${OpenCV_LIBS} pthread)
