//
// Created by wendy on 2021/6/10.
//
#ifndef INTEL_TASK_POOLING_H
#define INTEL_TASK_POOLING_H
#include "data.h"
#include "convolution.h"
#include <algorithm>
#include<immintrin.h>
#include <omp.h>
class PoolingLayer{
public:
    Data naive_maxpool2d(Data& input,int kernel_size,int *stride, int *padding);
    Data naive_avgpool2d(Data& input,int kernel_size,int *stride, int *padding);
    Data maxpool2d(Data& input,int kernel_size,int *stride, int *padding);
    Data maxpool2d1(Data& input,int kernel_size,int *stride, int *padding);
    Data avgpool2d(Data& input,int kernel_size,int *stride, int *padding);
};
#endif //INTEL_TASK_POOLING_H
