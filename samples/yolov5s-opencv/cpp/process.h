//
// Created by huaishan.yuan on 24-11-14.
//

#ifndef YOLOV5S_PROCESS_H
#define YOLOV5S_PROCESS_H

#include <iostream>
#include <vector>

namespace hm {
    typedef struct {
        float pos[4]; // x1, y1, x2, y2, normalize to 1.0
        float prob;
        float size;
        int type;
    } Bbox;

    typedef struct {
        int dlWidth;
        int dlHeight;
        int layerCnt;
        int anchorBoxCnt;
        int classNum;
        float confThreshold;
        float objThreshold;
        float nmsThreshold;

        float anchor[3][3 * 2];
        float scale[3];
        int w[3];
        int h[3];
        int c[3];
        int8_t *data[3];
    } Yolov5PostParam;

    int Yolov5PostProcessInt8(Yolov5PostParam *obj, std::vector<Bbox> &result);
} // :~ NS (hm)

#endif //YOLOV5S_PROCESS_H
