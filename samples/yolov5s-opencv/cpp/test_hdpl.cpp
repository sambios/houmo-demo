#include "hdpl/hdpl_runtime_api.h"
#include <iostream>
#include <assert.h>
#include <stdio.h>

int main(int argc ,char *argv[])
{

    int dev_count = 0;
    hdplGetDeviceCount(&dev_count);
    std::cout << "device count = " << dev_count << std::endl;
    hdplSetDevice(0);

    int test_size= (1<<10);
    void *devMem = nullptr;
    uint8_t *p1 = (uint8_t*)malloc(test_size);
    for(int i = 0; i < test_size; ++i) {
        p1[i] = i%255;
    }

    

    hdplMalloc(&devMem, test_size);
    hdplMemcpyHtoD(devMem, p1, test_size);

    uint8_t *p2 = (uint8_t*)malloc(test_size);
    hdplMemcpyDtoH(p2, devMem, test_size);

    for(int i = 0;i < test_size; ++i) {
        assert(p1[i] == p2[i]);
        //printf("%d ", p2[i]);
    }
    std::cout << "test OK!" << std::endl;

    return 0;
}