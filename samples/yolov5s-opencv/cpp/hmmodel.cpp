//
// Created by huaishan.yuan on 24-11-16.
//
#include "opencv2/opencv.hpp"
#include "base_model.h"

int main(int argc, char *argv[])
{
    const char *keys="{help | 0 | print help information.}"
                     "{@type t| info  | command type: info, test}"
                     //"{model | ../yolov5s_nohat.hmm | model file path}"
                     "{@models |../../models/yolov5s.hmm | modile file path}"
                     "{dev_id | 0 | device id}"
                     ;
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    std::string cmd = parser.get<std::string>("@type");
    std::cout << "cmd:" << cmd << std::endl;
    std::string model_file = parser.get<std::string>("@models");
    std::cout << "Input model:" << model_file << std::endl;
    int dev_id = parser.get<int>("dev_id");
    std::cout << "device id:" << dev_id << std::endl;

    hdplSetDevice(dev_id);
    hm::HMBaseModel baseModel(model_file);
    if (cmd == "info") {
        baseModel.PrintInfo(true);
    }else if (cmd == "test") {
        //pre process
        for(int i = 0; i< baseModel.InputTensorNum(); ++i) {
            //baseModel.InputTensor(0).Random();
        }

        baseModel.inference();
        //post process
    }
}