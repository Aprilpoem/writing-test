////
//// Created by wendy on 2021/6/9.
////
#include "convolution.h"
#include <iostream>
#include <ctime>
#include <cmath>
#include <omp.h>
//#include <x86intrin.h>
#include <xmmintrin.h>
#include <immintrin.h>
//#include <avxintrin.h>
using namespace std;
#define bs_i 256// l2 cache 256kb
#define bs_k 128
#define BlockSize 64

void print(float* data, int row, int col){
    for (int i = 0; i < row * col; i++){
        cout << data[i] << " ";
    }
    cout << endl;
}
/* im2col functions from DarkNet */
float im2col_get_pixel(Data &input, int batchID,
                       int row, int col, int channel, int padH, int padW){
    row -= padH;
    col -= padW;
    if (row < 0 || col < 0 ||
        row >= input.height || col >= input.width) return 0;

    return input.getValue(batchID,channel,row,col);
}

float* im2col(Data &input, int batchID, int padH, int padW, int strideH, int strideW, int ksize){
    int c,h,w;
    int height_col = (input.height + 2 * padH - ksize) / strideH + 1;
    int width_col = (input.width + 2 * padW - ksize) / strideW + 1;


    int channels_col = input.depth * ksize * ksize;
    float *data_col = new float[channels_col * height_col * width_col];
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * strideH;
                int im_col = w_offset + w * strideW;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(input,batchID,
                                                       im_row, im_col, c_im, padH, padW);
            }
        }
    }
    return data_col;
}
void gemm_nn_ikj(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
    //#pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                //C[i*ldc+j] += fma(A_PART,B[k*ldb+j],C[i*ldc+j]);
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void blocking(int M, int N, int K,
           float *A, int lda,
           float *B, int ldb,
           float *C, int ldc) {
#pragma omp parallel for
    for (int bi = 0; bi < M; bi++) {
        for (int bk = 0; bk < K; bk++) {
            __m256 a8 = _mm256_broadcast_ss(&A[bi * lda + bk]);
            for (int bj = 0; bj < N; bj += 8) {
                // Loads 8 SP FP values. The address must be 32-byte-aligned.
                __m256 b8 = _mm256_loadu_ps(&B[bk * ldb + bj]);
                __m256 c8 = _mm256_loadu_ps(&C[bi * ldc] + bj);
                c8 = _mm256_fmadd_ps(a8, b8, c8);
                // Store 8 values, address aligned
                _mm256_storeu_ps(&C[bi * ldc + bj], c8);
            }
        }
    }
}
void gemm_nn_ikj_blocking(int M, int N, int K, float ALPHA,
                 float *A, int lda,
                 float *B, int ldb,
                 float *C, int ldc){
    int i,k;
#pragma omp parallel for
    for(i = 0; i < M; i+=bs_i){
        for(k = 0; k < K; k+=bs_k){
                // do block
                blocking(min(M-i,bs_i),N,min(K-k,bs_k),&A[lda*i+k],lda,&B[ldb*k],ldb,&C[ldc*i],ldc);
        }
    }
}

void gemm_nn_ijk(int M, int N, int K, float ALPHA,
                 float *A, int lda,
                 float *B, int ldb,
                 float *C, int ldc)
{
    int i,j,k;
//#pragma omp parallel for private(j,k) schedule(dynamic)
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            for(k = 0; k < K; ++k){
                C[i*ldc+j] += A[i*lda+k]*B[k*ldb+j];
            }
        }
    }
}

void gemm_nn_ijk_blocking(int M, int N, int K, float ALPHA,
                 float *A, int lda,
                 float *B, int ldb,
                 float *C, int ldc)
{
    int i,j,k;
//#pragma omp parallel for private(j,k) schedule(dynamic)
    for(j = 0; j < N; j+=BlockSize){
            for( k = 0; k < K; k+=BlockSize) {
                for( i = 0; i < M; i++){
                    for (int bj = j; bj < min(N, j + BlockSize); bj++) {
                        for (int bk = k; bk < min(K, k + BlockSize); bk++) {
                            C[i * ldc + bj] += A[i * lda + bk] * B[bk * ldb + bj];
                        }
                    }
                }

            }
        }

}


void gemm_nn_ikj_simd(int M, int N, int K, float ALPHA,
                 float *A, int lda,
                 float *B, int ldb,
                 float *C, int ldc)
{
    int i,j,k;
#pragma omp parallel for private(i,k,j) schedule(dynamic)
    for(i = 0; i < M; i++){
        for(k = 0; k < K; k++){
           //Set all four words with the same value
            __m256 a4 = _mm256_set1_ps(A[i * lda + k]);
            for(j = 0; j < N; j+=8){
                // Loads 8 SP FP values. The address must be 32-byte-aligned.
                __m256 b4 = _mm256_loadu_ps(&B[k*ldb+j]);
                __m256 c4 = _mm256_loadu_ps(&C[i*ldc]+j);
                c4 = _mm256_fmadd_ps(a4,b4,c4);
                // Store 8 values, address aligned
                _mm256_storeu_ps(&C[i*ldc+j],c4);
            }
        }
    }
}


void gemm_relu(int M, int N, int K, float ALPHA,
                      float *A, int lda,
                      float *B, int ldb,
                      float *C, int ldc) {
    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < M; i++) {
        for (k = 0; k < K; k++) {
            __m256 a4 = _mm256_set1_ps(A[i * lda + k]);
            for(j = 0; j < N; j+=8){
                // Loads 8 SP FP values. The address must be 32-byte-aligned.
                __m256 b4 = _mm256_load_ps(&B[k*ldb+j]);
                __m256 c4 = _mm256_load_ps(&C[i*ldc]+j);
                c4 = _mm256_fmadd_ps(a4,b4,c4);
                // Store 8 values, address aligned
                _mm256_store_ps(&C[i*ldc+j],c4);
            }
        }
        for (j = 0; j < N; j+=8) {
            __m256 c8 = _mm256_load_ps(&C[i*ldc]+j);
            c8 = _mm256_max_ps(c8, _mm256_set1_ps(0.0));
            _mm256_storeu_ps(&C[i*ldc+j],c8);
        }
    }
}

ConvLayer::ConvLayer(int* padding,int* stride, int ksize, int in_depth, int out_depth){
    padding_h = padding[0];
    padding_w = padding[1];
    stride_h = stride[0];
    stride_w = stride[1];
    kernel_size = ksize;
    in_channel = in_depth;
    out_channel = out_depth;
    kernel.resetValue(out_channel,in_channel,kernel_size,kernel_size);
}

void ConvLayer::InitKernel(){
    kernel.RandomInit();
}

Data ConvLayer::naiveConv(Data &input){
    int outputH = computeShape(input.height,padding_h,stride_h,kernel_size);
    int outputW = computeShape(input.width,padding_w,stride_w,kernel_size);
    Data output = Data(input.batch,out_channel,outputH,outputW);
    output.Init(0);
    for(int idx=0; idx < input.batch; idx++) {
        for(int channel=0;channel < output.depth; channel++){
            for(int out_h=0; out_h < output.height; out_h++){
                for(int out_w=0; out_w < output.width; out_w++){
                    for(int ichannel=0; ichannel< input.depth; ichannel++){
                        for(int k_h=0; k_h < kernel.height;k_h ++){
                            for(int k_w=0;k_w <kernel.width; k_w++){
                                float tmpvalue = kernel.getValue(channel,ichannel,k_h,k_w) *
                                        input.getValue(idx,ichannel,out_h+k_h,out_w+k_w);
                                // 累加
                                output.AddValue(idx,channel,out_h,out_w,tmpvalue);
                            }
                        }
                    }

                }
            }
        }
    }

    //output.print();
    return output;
}

void ConvLayer::optimizedConv1(float *input,float *output,int batch,int m, int n, int k) const {
#pragma omp parallel for schedule(dynamic)
    for (int batch_id = 0; batch_id < batch; batch_id++) {
        // gemm to do convolution
        // kernel: m x k
        // input: k x n
        // output: m x n
        //cout << m << ' ' << n << ' ' << k << endl;
        gemm_nn_ikj_simd(m, n, k, 1.0,
                          kernel.data, k,
                          &input[batch_id * k * n], n,
                          &output[batch_id * m * n], n);
    }
}
void ConvLayer::optimizedConv(float *input,float *output,int batch,int m, int n, int k) const{
#pragma omp parallel for schedule(dynamic)
    for(int batch_id = 0; batch_id < batch; batch_id++){
        // gemm to do convolution
        // kernel: m x k
        // input: k x n
        // output: m x n
        //cout << m << ' ' << n << ' ' << k << endl;
        gemm_nn_ikj_blocking(m,n,k,1.0,
                kernel.data,k,
                &input[batch_id*k*n],n,
                &output[batch_id*m*n],n);
    }

//    for(int i=0;i< batch;i++){
//        for(int j=0;j< m*n;j++){
//            cout << output[i][j] << " ";
//        }
//        cout<<endl;
//    }

}
//
