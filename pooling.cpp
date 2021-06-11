//
// Created by wendy on 2021/6/10.
//
#include "pooling.h"
#include "data.h"
#include <algorithm>
#include <omp.h>
using namespace std;


Data PoolingLayer::maxpool2d(Data& input,int kernel_size, int *stride, int *padding) {
    int N = input.batch;
    int C = input.depth;
    int outputH = computeShape(input.height,padding[0],stride[0],kernel_size);
    int outputW = computeShape(input.width,padding[1],stride[1],kernel_size);
    Data output = Data(N,C,outputH,outputW);
    int i,j,outh,outw,m,n;
    #pragma omp parallel for private(j,outh,outw) schedule(dynamic)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < C; j++){
            for(int outh = 0; outh < outputH; outh++){
                for( int outw = 0; outw < outputW; outw++){
                    float tmp = 0;
//#pragma omp parallel reduction(max:tmp)
                    for( int m = 0; m < kernel_size; m ++){
                        for( int n = 0; n < kernel_size; n++){
                            tmp = max(tmp, input.getValue(i,j,stride[0]*outh+m,stride[1]*outw+n));
                        }
                    }
                    output.SetValue(i,j,outh,outw,tmp);
                }
            }
        }
    }
    return output;
}
Data PoolingLayer::avgpool2d(Data& input,int kernel_size, int *stride, int *padding) {
    int N = input.batch;
    int C = input.depth;
    int outputH = computeShape(input.height,padding[0],stride[0],kernel_size);
    int outputW = computeShape(input.width,padding[1],stride[1],kernel_size);
    Data output = Data(N,C,outputH,outputW);
#pragma omp parallel for collapse(4)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < C; j++){
            for(int outh = 0; outh < outputH; outh++){
                for( int outw = 0; outw < outputW; outw++){
                    float result = 0;
                    for( int m = 0; m < kernel_size; m ++){
                        for( int n = 0; n < kernel_size; n++){
                            result += input.getValue(i,j,stride[0]*outh+m,stride[1]*outw+n);
                        }
                    }
                    result = result / (float)(kernel_size * kernel_size);
                    output.SetValue(i,j,outh,outw,result);
                }
            }
        }
    }
    return output;
}

/* simple loop of pooling function*/
Data PoolingLayer::naive_maxpool2d(Data& input,int kernel_size, int *stride, int *padding) {
    int N = input.batch;
    int C = input.depth;
    int outputH = computeShape(input.height,padding[0],stride[0],kernel_size);
    int outputW = computeShape(input.width,padding[1],stride[1],kernel_size);
    Data output = Data(N,C,outputH,outputW);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < C; j++){
            for(int outh = 0; outh < outputH; outh++){
                for( int outw = 0; outw < outputW; outw++){
                    float tmp = 0;
                    for( int m = 0; m < kernel_size; m ++){
                        for( int n = 0; n < kernel_size; n++){
                            tmp = max(tmp, input.getValue(i,j,stride[0]*outh+m,stride[1]*outw+n));
                        }
                    }
                    output.SetValue(i,j,outh,outw,tmp);
                }
            }
        }
    }
    return output;
}
Data PoolingLayer::naive_avgpool2d(Data& input,int kernel_size, int *stride, int *padding) {
    int N = input.batch;
    int C = input.depth;
    int outputH = computeShape(input.height,padding[0],stride[0],kernel_size);
    int outputW = computeShape(input.width,padding[1],stride[1],kernel_size);
    Data output = Data(N,C,outputH,outputW);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < C; j++){
            for(int outh = 0; outh < outputH; outh++){
                for( int outw = 0; outw < outputW; outw++){
                    float result = 0;
                    for( int m = 0; m < kernel_size; m ++){
                        for( int n = 0; n < kernel_size; n++){
                            result += input.getValue(i,j,stride[0]*outh+m,stride[1]*outw+n);
                        }
                    }
                    result = result / (float)(kernel_size * kernel_size);
                    output.SetValue(i,j,outh,outw,result);
                }
            }
        }
    }
    return output;
}