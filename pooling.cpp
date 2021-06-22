//
// Created by wendy on 2021/6/10.
//
#include "pooling.h"
#include "data.h"
#include <algorithm>
#include <omp.h>
using namespace std;


Data PoolingLayer::maxpool2d1(Data& input,int kernel_size, int *stride, int *padding) {
    //__m256 _mm256_max_ps (__m256 a, __m256 b)
    int N = input.batch;
    int C = input.depth;
    int k = kernel_size;
    int outputH = computeShape(input.height,padding[0],stride[0],kernel_size);
    int outputW = computeShape(input.width,padding[1],stride[1],kernel_size);
    int n = outputW * outputH;
    Data outputs = Data(N,C,outputH,outputW);
    //float *output = new float[N*C*outputH*outputW]{0};
    float* output = (float *)_mm_malloc(N * C * n * sizeof(float), 32);
    fill(output,output+N*C*n,0.0);
    for(int batch_id = 0; batch_id < N; batch_id++){
        // do im2col to make data stored linear
        float *tmp = im2col(input, batch_id, padding[0], padding[1],
                               stride[0], stride[1], kernel_size);
        for(int c = 0;c < C;c++){
            for( int i=0;i < k * k ; i++){
                for( int j=0; j < n;j+=8){
                    __m256 inv = _mm256_loadu_ps(&tmp[(c*k*k+i)*n+j]);
                    __m256 outv = _mm256_loadu_ps(&output[(batch_id*C*n)+c*n+j]);
                    __m256 maxv = _mm256_max_ps(inv,outv);
                    _mm256_storeu_ps(output+(batch_id*C*n)+c*n+j,maxv);
                }
            }
        }
        //cout << output[batch_id*C*n] << endl;
    }
    outputs.setData(output);
    _mm_free(output);
    return outputs;
}

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
