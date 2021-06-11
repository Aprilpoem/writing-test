//
// Created by wendy on 2021/6/9.
//
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <algorithm>

#include "data.h"
#include "convolution.h"
#include "pooling.h"
#include "relu.h"
#include <omp.h>
#define LOOP_COUNT 5
using namespace std;


void testConv(Data &input,int kernel,int *stride, int *padding,int in_channel, int out_channel){
    ConvLayer convlayer = ConvLayer(padding,stride,kernel,in_channel,out_channel);
    convlayer.InitKernel();
    Data output;
    double starttime = omp_get_wtime();
    for( int i=0; i< LOOP_COUNT; i++){
        output = convlayer.naiveConv(input);
    }
    double endtime = omp_get_wtime();
    double time_avg = (endtime - starttime) / LOOP_COUNT;
    // naive conv has 7 loops, add and multiple
    double gflop = (2.0 * input.batch *  input.depth * output.depth * output.width * output.height * kernel * kernel)*1e-9;
    double gflops = gflop / time_avg;
    printf("Average time: %.5f secs \n", time_avg);
    printf("GFlop       : %.5f  \n", gflop);
    printf("GFlop/sec   : %.5f  \n", gflops);
}

void testOptimizedConv(Data &input,int kernel,int *stride, int *padding,int in_channel, int out_channel){
    ConvLayer convlayer = ConvLayer(padding,stride,kernel,in_channel,out_channel);
    convlayer.InitKernel();
    int batch = input.batch;
    int outputH = computeShape(input.height,padding[0],stride[0],kernel);
    int outputW = computeShape(input.width,padding[1],stride[1],kernel);
    int m = out_channel;
    int n = outputW * outputH;
    int k = kernel * kernel * in_channel;
    float** input_data = new float*[batch];
    float** output_data = new float*[batch];
    for(int batch_id = 0; batch_id < batch; batch_id++){
        // do im2col to make data stored linear
        float* tmpcol = im2col(input, batch_id, padding[0], padding[1], stride[0], stride[1], kernel);
        input_data[batch_id] = new float[ k*n];
        for(int i=0;i<k*n;i++){
            input_data[batch_id][i] = tmpcol[i];
        }
        output_data[batch_id] = new float[m * n]{0};
    }
    // compute conv time
    double starttime = omp_get_wtime();
    for( int i=0; i< LOOP_COUNT; i++){
        convlayer.optimizedConv(input_data,output_data, batch, m, n, k);
    }
    delete [] input_data;
    double endtime = omp_get_wtime();
    double time_avg = (endtime - starttime) / LOOP_COUNT;
    // naive conv has 7 loops, add and multiple
    double gflop = (2.0 * batch *  in_channel * out_channel * outputW * outputH * kernel * kernel)*1e-9;
    double gflops = gflop / time_avg;
    printf("optimized conv:");
    printf("Average time: %.5f secs \n", time_avg);
    printf("GFlop       : %.5f  \n", gflop);
    printf("GFlop/sec   : %.5f  \n", gflops);

}

void testPooling(Data &input, int kernel, int* stride,int* padding){
    PoolingLayer poolayer = PoolingLayer();
    int outputH = computeShape(input.height,padding[0],stride[0],kernel);
    int outputW = computeShape(input.width,padding[1],stride[1],kernel);

    // no openMP
    printf("maxpooling without openMP:");
    double starttime = omp_get_wtime();
    for( int i=0; i< LOOP_COUNT; i++){
        poolayer.naive_maxpool2d(input,kernel, stride, padding);
    }
    double endtime = omp_get_wtime();
    double time_avg = (endtime - starttime) / LOOP_COUNT;
    // naive conv has 7 loops, add and multiple
    double gflop = (2.0 * input.batch *  input.depth * outputW * outputH * kernel * kernel)*1e-9;
    double gflops = gflop / time_avg;
    printf("Average time: %.5f secs \n", time_avg);
    printf("GFlop       : %.5f  \n", gflop);
    printf("GFlop/sec   : %.5f  \n", gflops);

    // use openMP
    printf("maxpooling using openMP:");
    starttime = omp_get_wtime();
    for( int i=0; i< LOOP_COUNT; i++){
        poolayer.maxpool2d(input,kernel, stride, padding);
    }
    endtime = omp_get_wtime();
    time_avg = (endtime - starttime) / LOOP_COUNT;
    // naive conv has 7 loops, add and multiple
    gflop = (2.0 * input.batch *  input.depth * outputW * outputH * kernel * kernel)*1e-9;
    gflops = gflop / time_avg;
    printf("Average time: %.5f secs \n", time_avg);
    printf("GFlop       : %.5f  \n", gflop);
    printf("GFlop/sec   : %.5f  \n", gflops);

}

void testRelu(Data& input){
    double starttime = omp_get_wtime();
    for( int i=0; i< LOOP_COUNT; i++){
        relu(input);
    }
    double endtime = omp_get_wtime();
    double time_avg = (endtime - starttime) / LOOP_COUNT;
    double gflop = input.getSize()*1e-9;
    double gflops = gflop / time_avg;
    printf("relu:");
    printf("Average time: %.5f secs \n", time_avg);
    printf("GFlop       : %.5f  \n", gflop);
    printf("GFlop/sec   : %.5f  \n", gflops);
}
int main()
{

    // input data
    Data input = Data(10,3,100,100);
    input.RandomInit();
    //input.print();
    // kernel
    int padding[2] = {0,0};
    int stride[2] = {1,1};
    int kernel = 7;
    int in_channel = 3;
    int out_channel = 5;
    //testConv(input,kernel,stride,padding,in_channel,out_channel);
    testOptimizedConv(input,kernel,stride,padding,in_channel,out_channel);
    testPooling(input,kernel,stride,padding);
    testRelu(input);
    return 0;
}

