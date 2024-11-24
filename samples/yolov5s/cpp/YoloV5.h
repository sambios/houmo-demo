//
// Created by huaishan.yuan on 24-11-9.
//

#ifndef YOLOV5_H
#define YOLOV5_H

#include <iostream>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "base_model.h"
#include "process.h"

class YoloV5 : public hm::HMBaseModel {
    float m_confThreshold= 0.6;
    float m_nmsThreshold = 0.5;
    float m_objThreshold = 0.5;
    std::vector<std::string> m_coco_names;
    int m_minDim = 0;
    int m_classNum = 0;
public:
    explicit YoloV5(const std::string &model_file,
                    tcim::Device device = tcim::HDPL) : HMBaseModel(model_file, device) {

    }


    int SetParams(float conf = 0.5, float nms_threshold = 0.5,
                  float obj_threshold = 0.5, const std::string& coco_name_file="") {
        m_confThreshold = conf;
        m_nmsThreshold = nms_threshold;
        m_objThreshold = obj_threshold;

        return 0;
    }



    int Detect(const cv::Mat &image, std::vector<hm::Bbox> &objects) {

        //Preprocess
        auto inputTensor = InputTensor(0);
        assert(1 == inputTensor->BatchSize());
        // resize
        cv::Size inputSize(inputTensor->Width(), inputTensor->Height());
        cv::Mat resized_image;
        cv::resize(image, resized_image, inputSize);
        inputTensor->AllocHostMemory();
        inputTensor->AllocDeviceMemory();
        // RGB2YUV
        cv::Mat yuvImg;
        cv::cvtColor(resized_image, yuvImg, cv::COLOR_BGR2YUV_I420);
        hm::Utils::convertI420toNV12(yuvImg.data, (uint8_t*)inputTensor->HostMemoryPtr(), resized_image.cols, resized_image.rows);
        inputTensor->HostToDeviceSync();

        auto outputNum = OutputTensorNum();
        for (int i = 0; i < outputNum; ++i) {
            OutputTensor(i)->AllocHostMemory();
            OutputTensor(i)->AllocDeviceMemory();
        }

        //Inference
        inference();

        //Postprocess
        auto outputTensorShape = OutputTensor(0)->Shape();
        auto dims = outputTensorShape.size();
        //std::cout << ShapeToString(outputTensorShape) << std::endl;
        int cls_num = (int)outputTensorShape[dims - 1]/3 -5;

        hm::Yolov5PostParam postObj = {
                inputSize.width, inputSize.height,
                3, 3,
                cls_num,
                m_confThreshold, m_objThreshold, m_nmsThreshold,
                {
                        {10, 13, 16, 30, 33, 23},
                        {30, 61, 62, 45, 59, 119},
                        {116, 90, 156, 198, 373, 326},
                },
                {0.23858842253684998, 0.15234458446502686, 0.13460591435432434},
                {80, 40, 20},
                {48, 24, 12},
                {64, 64, 64}
        };

        for (int i = 0; i < outputNum; ++i) {
            OutputTensor(i)->DeviceToHostSync();
            postObj.data[i] = (int8_t*)OutputTensor(i)->HostMemoryPtr();
        }

        return hm::Yolov5PostProcessInt8(&postObj, objects);
    }
};


#endif //YOLOV5_H
