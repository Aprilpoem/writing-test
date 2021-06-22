//
// Created by wendy on 2021/6/9.
//
#include <iostream>
#include "data.h"
#include "convolution.h"
#include "pooling.h"
#include "relu.h"
#include <omp.h>
#include <immintrin.h>

#define LOOP_COUNT 5
#define alignedValue 32
using namespace std;

//typedef float  __attribute__((align_value (16))) float_align32;


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

int aligned_offset(int size, int alignment, int unit){
    // unit is 4byte
    alignment = alignment / unit;
    int offset = size % alignment;
    offset = (offset == 0) ? 0 : 8 - offset;
    return offset;
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

    int offset = aligned_offset(n,alignedValue,sizeof(float));
    int newn = n + offset;
    int input_size = k * newn;
    int output_size = m * newn;
    float *input_data = (float *)_mm_malloc(batch * input_size * sizeof(float), alignedValue);
    float *output_data = (float *)_mm_malloc(batch * output_size * sizeof(float),alignedValue);
    for(int batch_id = 0; batch_id < batch; batch_id++){
        // do im2col to make data stored linear
        float* tmpcol = im2col(input, batch_id, padding[0], padding[1], stride[0], stride[1], kernel);
        for(int i = 0;i < k*n;i++){
            input_data[batch_id*input_size+i] = tmpcol[i];
        }
        for(int i = 0;i < output_size;i++){
            output_data[batch_id*output_size+i] = 0.0;
        }
    }
    // compute conv time
    double starttime = omp_get_wtime();
    for( int i=0; i< LOOP_COUNT; i++){
        convlayer.optimizedConv(input_data,output_data, batch, m, newn, k);
    }
    double endtime = omp_get_wtime();
    double time_avg = (endtime - starttime) / LOOP_COUNT;
    // naive conv has 7 loops, add and multiple
    double gflop = (2.0 * batch *  in_channel * out_channel * outputW * outputH * kernel * kernel)*1e-9;
    double gflops = gflop / time_avg;
    printf("optimized conv:");
    printf("Average time: %.5f secs \n", time_avg);
    printf("GFlop       : %.5f  \n", gflop);
    printf("GFlop/sec   : %.5f  \n", gflops);
    _mm_free(input_data);
    _mm_free(output_data);



}

void testPooling(Data &input, int kernel, int* stride,int* padding){
    PoolingLayer poolayer = PoolingLayer();
    int outputH = computeShape(input.height,padding[0],stride[0],kernel);
    int outputW = computeShape(input.width,padding[1],stride[1],kernel);

    // no openMP
    printf("maxpooling without simd:");
    double starttime = omp_get_wtime();
    for( int i=0; i< LOOP_COUNT; i++){
        poolayer.maxpool2d(input,kernel, stride, padding);
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
    printf("maxpooling using simd:");
    starttime = omp_get_wtime();
    for( int i=0; i< LOOP_COUNT; i++){
        poolayer.maxpool2d1(input,kernel, stride, padding);
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


void testOptimizedConv1(Data &input,int kernel,int *stride, int *padding,int in_channel, int out_channel){
    ConvLayer convlayer = ConvLayer(padding,stride,kernel,in_channel,out_channel);
    convlayer.InitKernel();
    int batch = input.batch;
    int outputH = computeShape(input.height,padding[0],stride[0],kernel);
    int outputW = computeShape(input.width,padding[1],stride[1],kernel);
    int m = out_channel;
    int n = outputW * outputH;
    int k = kernel * kernel * in_channel;
    cout << m << ' ' << n << ' ' << k << endl;

    //int offset = aligned_offset(n,alignedValue,sizeof(float));
    int offset=0;
    int newn = n + offset;
    int input_size = k * newn;
    int output_size = m * newn;
    float *input_data = (float *)_mm_malloc(batch * input_size * sizeof(float), alignedValue);
    float *output_data = (float *)_mm_malloc(batch * output_size * sizeof(float),alignedValue);
    for(int batch_id = 0; batch_id < batch; batch_id++){
        // do im2col to make data stored linear
        float* tmpcol = im2col(input, batch_id, padding[0], padding[1], stride[0], stride[1], kernel);
        for(int i = 0;i < k*n;i++){
            input_data[batch_id*input_size+i] = tmpcol[i];
        }
        for(int i = 0;i < output_size;i++){
            output_data[batch_id*output_size+i] = 0.0;
        }
    }
    // compute conv time
    double starttime = omp_get_wtime();
    for( int i=0; i< LOOP_COUNT; i++){
        convlayer.optimizedConv(input_data,output_data, batch, m, newn, k);
    }
    double endtime = omp_get_wtime();
    double time_avg = (endtime - starttime) / LOOP_COUNT;
    // naive conv has 7 loops, add and multiple
    double gflop = (2.0 * batch *  in_channel * out_channel * outputW * outputH * kernel * kernel)*1e-9;
    double gflops = gflop / time_avg;
    printf("optimized conv with blocking:");
    printf("Average time: %.5f secs \n", time_avg);
    printf("GFlop       : %.5f  \n", gflop);
    printf("GFlop/sec   : %.5f  \n", gflops);
    for(int i=0; i<10;i++){
        cout << output_data[i] << ' ';
    }
    cout << endl;

    //no block
    fill(output_data,output_data+batch*output_size,0.0);
    double starttime1 = omp_get_wtime();
    for( int i=0; i< LOOP_COUNT; i++){
        convlayer.optimizedConv(input_data,output_data, batch, m, newn, k);
    }
    double endtime1 = omp_get_wtime();
    double time_avg1 = (endtime1 - starttime1) / LOOP_COUNT;
    // naive conv has 7 loops, add and multiple
    double gflop1 = (2.0 * batch *  in_channel * out_channel * outputW * outputH * kernel * kernel)*1e-9;
    double gflops1 = gflop / time_avg1;
    printf("optimized conv no blocking:");
    printf("Average time: %.5f secs \n", time_avg1);
    printf("GFlop       : %.5f  \n", gflop1);
    printf("GFlop/sec   : %.5f  \n", gflops1);
    for(int i=0; i<10;i++){
        cout << output_data[i] << ' ';
    }
    cout << endl;
}

void test_block(){
    int n = 500;
    int N = n * n;
    double gflop = 1e-9*(2.0*n*n*n);
    float *A =  new float [N];
    fill(A, A+N, 0.5);
    float *B = new float[N];
    fill(B, B+N, 2.5);
    float *C = new float [N];
    fill(C, C+N, 0.0);
    double starttime = omp_get_wtime();
    gemm_nn_ikj_simd(n,n,n,1.0,A,n,B,n,C,n);
    double endtime = omp_get_wtime();
    double gflops = gflop/(endtime-starttime);
    printf("GFlop/sec   : %.5f  \n", gflops);

    fill(C, C+N, 0.0);
    starttime = omp_get_wtime();
    gemm_nn_ikj_blocking(n,n,n,1.0,A,n,B,n,C,n);
    endtime = omp_get_wtime();
    gflops = gflop/(endtime-starttime);
    printf("GFlop/sec   : %.5f  \n", gflops);
}


void test_simd(){
    //int N,M;
    //cin >> N >> M;

    int m, n, k;
    m = 3;
    n = 21;
    k = 3;
    int S = 10;
    int alignment = 32;
    // 考虑到simd内循环数组起始地址和n有关(i*n)，因此n需要被8整除(n*4）%32 == 0
    int offset = n % 8;
    offset = (offset == 0) ? 0 : 8 - offset;
    int newn = offset + n;
    cout << newn << endl;
    int N = m * k; // 3x3
    int M = k * newn; // 3x16 = 48

    cout << N << " " << M << endl;
    float *A = (float *)_mm_malloc(N*sizeof(float),alignment);
    for (int i = 0; i < N; i++) {
        A[i] = i + 1;
    }
    float *B = (float *)_mm_malloc(S*M*sizeof(float),alignment);
    float *C = (float *)_mm_malloc(S*M*sizeof(float),alignment);
    //float* __attribute__((aligned(32))) C = new float(24);
    //float C[N][M] __attribute__((aligned (32)));// m x n 3x8=24
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < M; j++) {
            B[i*M+j] = 1.0;
            C[i*M+j] = 0.0;
        }
    }
    for(int batch_id = 0; batch_id < S; batch_id++){
        gemm_nn_ikj_simd(m,newn,k,1.0,
                         A,k,
                         &B[batch_id*M],newn,
                         &C[batch_id*M],newn);
    }
    for (int i = 0; i < S; i++) {
        for (int ii = 0; ii < m; ii++) {
            for( int j = 0; j < n;j++)
            cout << C[i*M+ii*newn+j] << ' ';
            cout << endl;
        }
        cout << endl;
    }

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
    //test_simd();
    //testConv(input,kernel,stride,padding,in_channel,out_channel);
    testOptimizedConv1(input,kernel,stride,padding,in_channel,out_channel);
    //testPooling(input,kernel,stride,padding);
    //testRelu(input);
    //test_block();
    //testAlignedData(input,kernel,stride,padding,in_channel,out_channel);
    return 0;
}

