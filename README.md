# HOUMO demos

目前提供比较流行的样例如YOLO系列的，后面逐步添加支持。

2025-4-1 更新日志

*   新增YOLOv5s例子代码
*   新增YOLOv8m离子代码

## 一、依赖版本说明

后摩大道版本: 2.1.0, 请从官网下载版本，[https://developer.houmoai.com](https://developer.houmoai.com/)

> 注意：下载后摩大道的软件包，需要您注册成为VIP用户，请注册后联系我们。

下载完成之后，安装驱动可以执行如下指令：

```shell
sudo bash ./houmo_drv_v2.1.0_ubuntu2004_x86_64.run install all 
tar zxf LM_VPU_fw_v2.1.0.tar.gz 
cd LM_VPU_fw 
# 假设你的设备是/dev/dri/renderD128​
# 对于M30, loader.img 先不要升级。
# sudo hmupdate-tools-v3.2.7-linux-aarch64/hmupdate -f ./loader.img -d /dev/dri/renderD130
# 更新firmware​
sudo hmupdate-tools-v3.2.7-linux-aarch64/hmupdate -f ./boot.img -d /dev/dri/renderD130
# 烧写完成之后，建议将机器重新启动
```



## 二、编译方法

### 2.1 编译之前您需要根据您自己的环境设置一下相关的环境变量

```bash
# 设置TCIM解压的路径,根据你解压的真实路径来修改
export TCIM_RUNTIME_PATH=/path/to/houmo-tcim-runtime
# 设置环境
source scripts/envsetup.sh
```

### 2.2 编译程序

```bash
cd samples
mkdir build && cd build
cmake .. && make -j4
```

### 2.3 运行程序

    $build 假设当前在build目录下
    build$ ./bin/yolov5 ../yolov5/yolov5s.hmm ../yolov5/bus.jpg 
    # 如果您的及其没有后摩卡，需要使用模拟器运行，否则会出现如下错误：

    ===> yolov5s c++ example start...
    tcim version: 2.1.0 Mar 14 2025
    LoadFromFile yolov5s
    [HM_HAL][E][hm800_drm_device_getinfo][L:266]No hm800 card found under path: /dev/dri
    [HM_HAL][E][hm800_drm_get_device_info][L:156]failed to get device info
    yolov5: /home/workspace/hm_sdk/misc/host/runtime/src/ipu/ipu_driver.cpp:11: ipu::IpuDriver::IpuDriver(): Assertion `false && "Failed to find hm800 card!"' failed.
    已中止 (核心已转储)
    # 可以通过如下方法解决
    export HDLP_PLATFORM=ISIM
    # 正确运行应该有如下的类似输出
    ===> yolov5s c++ example start...
    tcim version: 2.1.0 Mar 14 2025
    LoadFromFile yolov5s
    model ../yolov5/yolov5s.hmm loaded.
    Count of Input: 1
    Input[images] TensorInfo: shape: [1,3,640,640], stride: [614400,640,1,614400,640,2,1,409600], dtype: UINT8, format: YUV420SP, size: 614400, memsize: 614400
    Count of Output: 3
    Output[340] TensorInfo: shape: [1,3,80,80,85], stride: [], dtype: FLOAT32, format: ND, size: 6528000, memsize: 6528000
    Output[378] TensorInfo: shape: [1,3,40,40,85], stride: [], dtype: FLOAT32, format: ND, size: 1632000, memsize: 1632000
    Output[416] TensorInfo: shape: [1,3,20,20,85], stride: [], dtype: FLOAT32, format: ND, size: 408000, memsize: 408000

    ******************************
    *                            *
    *  KERNEL RUN ON ISIM HM800  *
    *                            *
    ******************************

    detect num: 5
    box[129, 241, 207, 523], conf:0.812449, cls:0
    box[392, 229, 479, 526], conf:0.799451, cls:0
    box[9, 129, 474, 473], conf:0.776166, cls:5
    box[32, 239, 127, 538], conf:0.774025, cls:0
    box[0, 333, 41, 516], conf:0.384387, cls:0
    demo results saved to demo_results/bus.jpg
    <=== yolov5s c++ example completed.

### 2.4 关于模型的说明

*   yolov5s模型信息如下：

> ```shell
> Input[images] TensorInfo: shape: [1,3,640,640], stride: [614400,640,1,614400,640,2,1,409600], dtype: UINT8, format: YUV420SP, size: 614400, memsize: 614400
> Count of Output: 3
> Output[340] TensorInfo: shape: [1,3,80,80,85], stride: [], dtype: FLOAT32, format: ND, size: 6528000, memsize: 6528000
> Output[378] TensorInfo: shape: [1,3,40,40,85], stride: [], dtype: FLOAT32, format: ND, size: 1632000, memsize: 1632000
> Output[416] TensorInfo: shape: [1,3,20,20,85], stride: [], dtype: FLOAT32, format: ND, size: 408000, memsize: 408000
> ```

*   yolov8m的模型信息如下：

```shell
Input [images] :TensorInfo: shape: [1,3,640,640], stride: [614400,640,1,614400,640,2,1,409600], dtype: UINT8, format: YUV420SP, size: 614400, memsize: 614400
Count of Output: 1
Output[output0] TensorInfo: shape: [1,84,8400], stride: [], dtype: FLOAT32, format: ND, size: 2822400, memsize: 2822400
```
参考模型下载地址：链接: https://pan.baidu.com/s/1MUNcNGi9llDSzEtEN4f1Ng?pwd=ddig 提取码: ddig 


