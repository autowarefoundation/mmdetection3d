## Introduction

The **[mmdetection3d](https://github.com/open-mmlab/mmdetection3d)** repository includes an additional voxel encoder
feature for the CenterPoint 3D object detection model, known as voxel center z,
not originally used in the **[main implementation](https://github.com/tianweiy/CenterPoint)**,
Autoware maintains consistency with the input size of the original implementation. Consequently,
to ensure integration with Autoware's lidar centerpoint package, we have forked the original repository and made
the requisite code modifications.

To train custom CenterPoint models and convert them into ONNX format for deployment in Autoware, please refer to the instructions provided in the README.md file included with
Autoware's **[lidar_centerpoint](https://autowarefoundation.github.io/autoware.universe/main/perception/lidar_centerpoint/)** package. These instructions will provide a step-by-step guide for training the CenterPoint model.
