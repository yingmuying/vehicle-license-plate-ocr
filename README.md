# 车辆检测&车牌检测&车牌识别系统

## 模块简介

* 车辆检测![carA.png](samples%2FcarA.png)
* 车牌检测![carA_lp.png](samples%2FcarA_lp.png)
* demo视频[111.mp4](111.mp4)

### 车辆检测

* 使用 yolov4
  进行车辆检测，开源代码[Darknet project website](https://github.com/AlexeyAB/darknet#how-to-improve-object-detection).

### 车牌检测

* 基于 tensorflow 卷积神经网络实现车牌检测、矫正.

### 车牌识别

* 基于 darknet 训练 35 类别的检测网络(25个字母和10个数字，O和0认为是一个)实现车牌识别.

## 配置说明

* 编译 darknet: 根据 GPU、CUDA、opencv 情况修改`Makefile`,然后 make (cd darknet && make).
* python3.7 环境 requirements.txt

## 使用说明

* vehicle_detect.py 实现车辆检测，调用 darkent.py.
* lp_detect_align_tf.py 实现车牌检测和矫正.
* lp_ocr.py 实现车牌识别.
* main.py 实现整体系统：车辆检测、车牌检测、车牌识别.


