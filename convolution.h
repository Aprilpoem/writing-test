//
// Created by wendy on 2021/6/9.
//
#include "data.h"
#ifndef INTEL_TASK_CONVOLUTION_H
#define INTEL_TASK_CONVOLUTION_H
class ConvLayer{
public:
    int padding_h;
    int padding_w;
    int stride_h;
    int stride_w;
    int kernel_size;
    int out_channel;
    int in_channel;
    Data kernel;
    // function
    ConvLayer(int *padding,int *stride, int kernel, int in_depth, int out_depth);
    void InitKernel();
    Data naiveConv(Data &input);
    //Data ConvCpu(Data &input);
    void optimizedConv(float **input, float** output,int batch,int m, int n, int k);

};
float* im2col(Data &input, int batchID, int padH, int padW, int strideH, int strideW, int ksize);
#endif //INTEL_TASK_CONVOLUTION_H
