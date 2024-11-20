//
// Created by huaishan.yuan on 24-11-16.
//

#include "process.h"
#include <algorithm>
#include <cmath>
#include "opencv2/opencv.hpp"

namespace hm {

    inline float sigmoid(float x) {
        return 1.f / (1.f + expf(-x));
    }

    inline float calc_iou(Bbox &a, Bbox &b) {
        float in_w = MIN(a.pos[2], b.pos[2]) - MAX(a.pos[0], b.pos[0]);

        if (in_w > 0) {
            float in_h = MIN(a.pos[3], b.pos[3]) - MAX(a.pos[1], b.pos[1]);
            if (in_h > 0) {
                float iou = in_w * in_h;
                return iou / (a.size + b.size - iou);
            }
        }
        return 0;
    }

    void NMS(std::vector<Bbox> &input, std::vector<Bbox> &output, float nms_thread) {
        std::sort(input.begin(), input.end(),
                  [](const Bbox &a, const Bbox &b) {
                      return a.prob == b.prob ? a.size > b.size : a.prob > b.prob;
                  }
        );

        for (int i = 0; i < input.size(); i++) {
            if (input[i].prob <= 0) {
                continue;
            }

            for (int j = i + 1; j < input.size(); j++) {
                if (input[j].prob > 0) {
                    float iou = calc_iou(input[i], input[j]);
                    if (iou >= nms_thread) {
                        input[j].prob = 0;
                    }
                }
            }
        }

        for (auto &in: input) {
            if (in.prob > 0) {
                output.push_back(in);
            }
        }
    }

    int Yolov5PostProcessInt8(Yolov5PostParam *obj, std::vector<Bbox> &result) {
        int i = 0, j = 0, k = 0, l = 0;
        int8_t *px = NULL;
        int8_t *py = NULL;
        int8_t *pw = NULL;
        int8_t *ph = NULL;
        int8_t *obj_score_ptr = NULL;
        int8_t *class_score_ptr = NULL;
        float obj_score, class_score;
        int channelLen = 0;
        Bbox tmpBox;
        std::vector<Bbox> boxVec;

        boxVec.resize(0);
        result.resize(0);

        for (i = 0; i < obj->layerCnt; i++) {
            channelLen = obj->w[i] * obj->h[i];
            for (k = 0; k < channelLen; k++) {
                for (j = 0; j < obj->anchorBoxCnt; j++) {
                    px = obj->data[i] + k * obj->c[i] + j * (obj->classNum + 5);
                    py = px + 1;
                    pw = py + 1;
                    ph = pw + 1;
                    obj_score_ptr = ph + 1;
                    class_score_ptr= obj_score_ptr + 1;
                    obj_score = sigmoid((*obj_score_ptr) * obj->scale[i]);
                    if (obj->objThreshold > obj_score) {
                        continue;
                    }

                    auto max_score_itor = std::max_element(class_score_ptr, class_score_ptr + obj->classNum);
                    int8_t class_max_score_int8 = *max_score_itor;
                    auto type = std::distance(class_score_ptr, max_score_itor);

                    float inv_w = 1.f / obj->w[i];
                    float inv_h = 1.f / obj->h[i];
                    float inv_aw = obj->anchor[i][j * 2] / obj->dlWidth;
                    float inv_ah = obj->anchor[i][j * 2 + 1] / obj->dlHeight;

                    tmpBox.prob = sigmoid(class_max_score_int8 * obj->scale[i]) * obj_score;
                    if (tmpBox.prob >= obj->confThreshold) {
                        tmpBox.type = type;
                        tmpBox.pos[0] = (k % obj->w[i] + (2 * sigmoid((*px) * obj->scale[i]) - 0.5)) * inv_w;
                        tmpBox.pos[1] = (k / obj->w[i] + (2 * sigmoid((*py) * obj->scale[i]) - 0.5)) * inv_h;
                        tmpBox.pos[2] = powf((2.f * sigmoid((*pw) * obj->scale[i])), 2) * inv_aw;
                        tmpBox.pos[3] = powf((2.f * sigmoid((*ph) * obj->scale[i])), 2) * inv_ah;
                        tmpBox.size = tmpBox.pos[2] * tmpBox.pos[3];
                        tmpBox.pos[0] -= tmpBox.pos[2] * 0.5;
                        tmpBox.pos[1] -= tmpBox.pos[3] * 0.5;
                        tmpBox.pos[2] += tmpBox.pos[0];
                        tmpBox.pos[3] += tmpBox.pos[1];
                        boxVec.push_back(tmpBox);
                    }
                }
            }
        }

        NMS(boxVec, result, obj->nmsThreshold);

        return 0;
    }
}//:~ NS(hm)