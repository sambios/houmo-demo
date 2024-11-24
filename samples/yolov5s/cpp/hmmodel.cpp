//
// Created by huaishan.yuan on 24-11-16.
//
#include "opencv2/opencv.hpp"
#include "base_model.h"
#include "utils.h"

int main(int argc, char *argv[])
{
    const char *keys="{help | 0 | print help information.}"
                     "{@type t| info  | command type: info, test}"
                     //"{model | ../yolov5s_nohat.hmm | model file path}"
                     "{model m |../../models/yolov5s-v7.0.hmm | modile file path}"
                     "{dev_id | 0 | device id}"
                     ;
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    std::string cmd = parser.get<std::string>("@type");
    std::cout << "cmd:" << cmd << std::endl;
    std::string model_file;
    if (parser.has("model")) {
        model_file = parser.get<std::string>("model");
        std::cout << "Input model:" << model_file << std::endl;
    }

    int dev_id = parser.get<int>("dev_id");
    std::cout << "device id:" << dev_id << std::endl;

    hdplSetDevice(dev_id);
    hm::HMBaseModel baseModel(model_file);
    if (cmd == "info") {
        baseModel.PrintInfo(true);
    }else if (cmd == "test") {
        TimeStamp ts;
        LOG_TS(&ts, "preprocess time");
        //pre process
        for(int i = 0; i< baseModel.InputTensorNum(); ++i) {
            baseModel.InputTensor(0)->GenData(true);
            baseModel.InputTensor(0)->AllocDeviceMemory();
            baseModel.InputTensor(0)->HostToDeviceSync();
        }
        LOG_TS(&ts, "preprocess time");

        LOG_TS(&ts, "forward time");
        baseModel.inference();
        LOG_TS(&ts, "forward time");

        //post process
        LOG_TS(&ts, "post process time");
        for(int i = 0; i < baseModel.OutputTensorNum(); ++i) {
            baseModel.OutputTensor(0)->AllocHostMemory();
            baseModel.OutputTensor(0)->DeviceToHostSync();
        }
        LOG_TS(&ts, "post process time");

        ts.build_timeline("hm_model");
        ts.show_summary("hm_model");
    }
}