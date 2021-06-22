# Revised part
To optimize gemm, I add two parts: 1. In Simd, I use `_mm256xx` to replace `_mm128xxx`, so gemm runs faster. 1. Besides, I add Block to reduce cache misses. And I apply simd to `maxpooling` and fused `conv+relu`. Some details described as following:

- Beacuse avx2 support 256 Bytes data, so in code, we can get `8 floats` as vector.

  - The address is not 32-byte-aligned.

    Use `_mm256_loadu_ps` and `_mm256_storeu_ps` to load and store data whose address are not 32-byte-aligned.

    ```c++
    for(i = 0; i < M; i++){
            for(k = 0; k < K; k++){
               //Set all four words with the same value
                __m256 a4 = _mm256_set1_ps(A[i * lda + k]);
                for(j = 0; j < N; j+=8){
                    __m256 b4 = _mm256_loadu_ps(&B[k*ldb+j]);
                    __m256 c4 = _mm256_loadu_ps(&C[i*ldc]+j);
                    c4 = _mm256_fmadd_ps(a4,b4,c4);
                    // store
                    _mm256_storeu_ps(&C[i*ldc+j],c4);
                }
            }
        }
    ```

  - The address is  32-byte-aligned.

    Use `_mm_malloc(size_t __size, size_t __align)` to allocate memory and `_mm_free` to free it.

    ```c++
    #define alignedValue 32
    float *input_data = (float *)_mm_malloc(batch * input_size * sizeof(float), alignedValue);
    float *output_data = (float *)_mm_malloc(batch * output_size * sizeof(float),alignedValue);
    ```

    Because when we use `_mm256_load_ps(float *addr)`, the address must be 32-byte-aligned. As we load 8 floats at once ( 8 * sizeof(float) = 32 ), if start address is aligned,  then interest loop `j` can keep address aligned. So we need to make sure the row address ( start address of `j` loop )of matrix should be 32-byte-aligned. The number of columns of Matix is a multiple of `8`. ( (columns * sizeof(floats)) % 32 == 0)

    ```c++
    int aligned_offset(int size, int alignment, int unit){
        // unit is 4byte
        alignment = alignment / unit;
        int offset = size % alignment;
        offset = (offset == 0) ? 0 : alignment - offset;
        return offset;
    }
    int offset = aligned_offset(n,alignedValue,sizeof(float));
    int newn = n + offset;
    int input_size = k * newn;
    int output_size = m * newn;
    float *input_data = (float *)_mm_malloc(batch * input_size * sizeof(float), alignedValue);
    float *output_data = (float *)_mm_malloc(batch * output_size * sizeof(float),alignedValue);
    ```

  - The performace of aligned address is higher , the Gflops is `7.51`, while the Gflops is `7.18` when not align the address of data.

    

- Use blocking to accelerate computation.

The goal of using `blocking` is to maximize data reuse before it is replaced in cache. 

Different level cache sizes of my laptop are shown below:

```c++
hw.cachelinesize: 64 = 64 B
hw.l1icachesize: 32768 = 32 KB
hw.l1dcachesize: 32768 = 32 KB
hw.l2cachesize: 262144 =  256 KB
hw.l3cachesize: 6291456 = 6 GB
```

We choose `bs_i x bs_k` sub-matrix from A, and `bs_k x N` from matrix B to get sub-matrix C, which size is `bs_i x N`. Note that `[bs_i x  N]` sub-matrix C not finished, it should computed K/bk loops. 

```c++
#define bs_i 256
#define bs_k 256 //l2 cache 256kb, it could load 256x256 floats
void blocking(int M, int N, int K,
           float *A, int lda,
           float *B, int ldb,
           float *C, int ldc) {
#pragma omp parallel for
    for (int bi = 0; bi < M; bi++) {
        for (int bk = 0; bk < K; bk++) {
            __m256 a8 = _mm256_broadcast_ss(&A[bi * lda + bk]);
            for (int bj = 0; bj < N; bj += 8) {
                __m256 b8 = _mm256_loadu_ps(&B[bk * ldb + bj]);
                __m256 c8 = _mm256_loadu_ps(&C[bi * ldc] + bj);
                c8 = _mm256_fmadd_ps(a8, b8, c8);
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
```

Simply use N rows and N columns matrix multiplation to test blocking algorithm. 

```c++
void test_block(){
    int n = 1024;
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
```

|                        | n=50 | n=100 | n=500 | n=1024 |
| ---------------------- | ---- | ----- | ----- | ------ |
| `gemm_nn_ikj_simd`     | 0.6  | 2.78  | 7.7   | 9.85   |
| `gemm_nn_ikj_blocking` | 1.15 | 1.38  | 6.0   | 26.16  |

From this table we can find the performance is related to the size of matrix. If matrix could be load cache entirly,  then the blocking is not working, even it could slow the speed because of additional operations of dividing matrix. But if the matrix is big enough ( I don't know exactly how to determine this boundary ), the experiments show when n > 500, the blocking algorithm works.

- Apply simd to pooling function and fused conv+relu function.

  - if use simd to pooling function,  we can also use `im2col` to reshape input data, and so that we can use `_mm256_max_ps(__m256 a, __m256 b)`compare 8 floats at once. 

    In `k^2` loop, we can get `8` results  ( k x k) .

    ```c++
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
        }
        outputs.setData(output);
        _mm_free(output);
        return outputs;
    }
    ```

    

  - add simd to cone_relu

    ```c++
    void gemm_relu(int M, int N, int K, float ALPHA,
                          float *A, int lda,
                          float *B, int ldb,
                          float *C, int ldc) {
        int i, j, k;
    #pragma omp parallel for
        for (i = 0; i < M; i++) {
            for (k = 0; k < K; k++) {
                .....
            }
            // load 8 floats at once to do x=max(0,x)
            for (j = 0; j < N; j+=8) {
                __m256 c8 = _mm256_load_ps(&C[i*ldc]+j);
                c8 = _mm256_max_ps(c8, _mm256_set1_ps(0.0));
                _mm256_storeu_ps(&C[i*ldc+j],c8);
            }
        }
    }
    ```

    

  

  ---

# Report for intel writing test

Mail: wulsh26@mail2.sysu.edu.cn

Author: Longshi Wu

In this test,  I use c++ to implement convolution , relu and pooling functions.  

## To do

- [x] Conv2D, Pooling2D and RELU operator
- [x] Forward
- [ ] Backward
- [x] Optimization skill
  - [x] SIMD
  - [x] OpenMP
  - [x] Cache locality
- [x] Advantage feature
  - [ ] Fuse Conv + relu + pooling into one function
  - [x] Parallel three functions with OpenMP



## Prerequisity

### Laptop CPU Specs

Using command line `sysctl -a | grep cpu` to check those information below.

- Hardware：Intel® Core™ i5-8257U Processor @1.4GHz 4 core， support AVX2, 2 FMA, SSE2 

- Operating System：macOS version 10.14.6 (18G103)

So the **peak GFLOPs** = 4 x 1.4 x 32 = 179.2GFlOPs in this machine.

Similarly, for a single core, this number is 44.8 GFLOPs.



### Running

```
g++ main.cpp convolution.cpp data.cpp relu.cpp pooling.cpp -fopenmp -mfma -mavx
./a.out
```

Make sure `openMP` installed, then use this two command lines to run this simple pipeline.



### Input Data 

Because this work now only inlcude forward pass part,  we can just input some test data into those functions ( conv2d, relu, pooling ) to compute performance ( flops ).  Thus, The test data temporarily don't need labels, which can be generated randomly.

Assume the input data shape is $[N,C_{in},H_{in},W_{in}]$. The multi-dimentional array is a little hard to construct using c++, as arrays are physically stored in a linear, one-dimentional computer memory, I just using one-dimentional array to store data. So for input data $[N,C_{in},H_{in},W_{in}]$, the length of one-dimentional array is $N*C_{in}*H_{in}*W_{in}$. Similarly，the kernel data whose size is $[C_{out},C_{in},KH,KW]$ has length equal to $C_{out}*C_{in}*KH*KW$ . The output data has shape $[N,C_{out},H_{out},W_{out}]$, length $[N*C_{out}*H_{out}*W_{out}]$.

**Data Structure **

```c++
class Data{
public:
    int batch,depth,width,height;
    float *data;
    // function
    Data();
    Data(int n,int c, int h, int w);
    ~Data();
    Data& operator=(const Data & chunks);
    float getValue(int i, int j, int m, int n);
    void SetValue(int i, int j, int m, int n, float  value);
    void AddValue(int i, int j, int m, int n, float value);
    void RandomInit();
    void Init(float fillValue = 0);
    void print();
};
```

### Measure the Performance

Here，we use flops to measure the performance of functions like convolution, relu and pooling.  Because the number of operations  inside the whole program is hard to calculate，here use the number of operations of matrix multiplication. Addtionally, we use `omp_get_wtime()` to measure the system time, which return the number of seconds. The code of computing flops are showed as follows:

```c++
#define LOOP_COUNT 5
double starttime = omp_get_wtime();
for( int i=0; i< LOOP_COUNT; i++){
  // funtion: conv,relu,pooling
  ...
}
double endtime = omp_get_wtime();
// average time 
double time_avg = (endtime - starttime) / LOOP_COUNT;
// compute the number of operations
double gflop = (2.0 * N * C_in * C_out * H_out * W_out * KH * KW)*1e-9;
// compute flops
double gflops = gflop / time_avg;
```



## Convolution

### Simple for-loop convolution

The direct convolution is simple for-loop convolution, call it as `naiveConv`.

```c++
for(int idx=0; idx < input.batch; idx++) {
    for(int channel=0;channel < output.depth; channel++){
        for(int out_h=0; out_h < output.height; out_h++){
            for(int out_w=0; out_w < output.width; out_w++){
                for(int ichannel=0; ichannel< input.depth; ichannel++){
                    for(int k_h=0; k_h < kernel.height;k_h ++){
                        for(int k_w=0;k_w <kernel.width; k_w++){
```

Clearly, it has 7 nested for loops. Using naiveConv as basline to compare with some optimized methods.

### Optimized convolution

Implment the optimized convolution function follow the commonly used optimized method: im2col + GEMM . 

#### im2col (image to column)

To convert for-loop convolution to maxtrix multiplication, we need to transfrom input image to matrix. In for-loop convolution,  the input patch data where the conv filter is applied actually not stored linearly, which slow the speed of calculation.



 **im2col** is proposed to rerange input data to make data stored linearly, then cpu can access data faster. The key idea is to find all the patches traversed by the convolution kernel in sequence  at one time, and reorder those patches.

#### GEMM ( generalized maxtrix multiplication)

As we know that matrix product could get the conv output directly (as below formula shows ) , then the key point now is how to accelerate GEMM. 
$$
C_{M \times N}+= A_{M \times K} * B_{K \times N}
$$
Here I use three skills. 

- **Loop reordering**.  Reorder the loops to access data efficiently.
- Using **openMP** to  parallelize outer loop.
- Using **SIMD** to do vectorization. It processes multiple data streams using a single instruction stream to accelerate computation.

```c++
/*GEMM function */
void gemm_nn_ijk(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc);
void gemm_nn_ikj(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc); 
void gemm_nn_ikj_simd(int M, int N, int K, float ALPHA,
                 float *A, int lda,
                 float *B, int ldb,
                 float *C, int ldc)
```

- **Loop Reordering**

```c++
/* loop order is i,j,k */
void gemm_nn_ijk(...){
  ...
  for(i = 0; i < M; ++i){
          for(j = 0; j < N; ++j){
              for(k = 0; k < K; ++k){
                  C[i*ldc+j] += A[i*lda+k]*B[k*ldb+j];
              }
          }
  }
}
```

The last inner loop is traverse `k` from `0` to `K` , so when find `B[k,j]`, the next element `B[k+1,j] `may not cached, it need time to access data from RAM. So if we

reorder the loops from `i,j,k` to `i,k,j`, it may reduce cache misses.

```c++
/* loop order is i,k,j */
for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
```

- **Threading**

Using openMP to  parallelize loops.

```c++
#include <omp.h>
// add this directive before outer loop
#pragma omp parallel for private(k,j) schedule(dynamic) 
for(i = 0; i < M; ++i){
  ...
```

- **SIMD**

Single Instruction Multiple Data, or SIMD, which can be used to do the same operation( add, multiply, etc .) on multiple values simultaneously. A single-precision floating-point number occupies 4 bytes, the laptop I used support AVX2, so it could operate four floating-point numbers at the same time ( 128-bit vector containing **4** `floats`). 





```c++
#include <x86intrin.h>
void gemm_nn_ikj_simd(...){
  ...
for(i = 0; i < M; i++){
        for(k = 0; k < K; k++){
            __m128 a4 = _mm_set1_ps(A[i*lda+k]);
            for(j = 0; j < N; j+=4){ // here vectorized 4 float
                __m128 b4 = _mm_load_ps(&B[k*ldb+j]);
                __m128 c4 = _mm_load_ps(&C[i*ldc]+j);
                c4 = _mm_add_ps(_mm_mul_ps(a4,b4),c4);
                _mm_store_ps(&C[i*ldc+j],c4);
            }
        }
    }
}
```

### Testing

> Input shape: [ 10 x 3 x 100 x 100]
>
> Kernel shape: [ 5 x 3 x 7 x 7 ], padding = [0,0], stride=[1,1]
>
> Output shape:  [ 10 x 5 x 94 x 94]

|                          | GLOPS  |
| :----------------------: | ------ |
| ``naiveConv`` (baseline) | 0.1078 |
|      `gemm_nn_ijk`       | 1.3126 |
|      `gemm_nn_ikj`       | 2.3871 |
|    `gemm_nn_ikj_simd`    | 3.672  |

As we mentioned before, `naiveConv` has simple 7 nested loops, `gemm_nn_xxx` all used `im2col` and OpenMP to parallelize loops. The last one, `gemm_nn_ikj_simd`, used all three optimized skill in the context , which increased by almost 36 times compared with baseline ( 3.6 vs 0.1).

## Pooling

- Implemented functions: `maxpool2d` and `avgpool2d`
- no learnable parameters
- use openMP to parallelize loops.

```c++
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
```

- Testing
  - Input data and kernel size have same shape as the convolution test case. 
  - `avgpool2d`  **0.24 Gflops** /**0.60 Gflops**. ( no parallel / with parallel )
  - `avgpool2d`  **0.15 Gflops** /**0.57 Gflops**. ( no parallel / with parallel )

## ReLu

$$
ReLU(x)=max(0,x)
$$

`relu` function is easy to implement, as it only need to compared itself with zero. The element-wise operate can use `OpenMP` to parallize. Because it has no paremeters, we can fuse convolution and relu to reduce runtime.

```c++
// void gemm_relu()
for(i = 0; i < M; i++){
	for(k = 0; k < K; k++){
      for(j = 0; j < N; j+=4){...}
  }
  // add this line to implement relu in-palce
	for (j = 0; j < N; ++j) {
    C[i*ldc+j] = ( C[i*ldc+j] > 0) ? C[i*ldc+j]:0;
```



## Summary

This test was very interesting and challenging for me. In the process of writing code while searching information, I learned  that the implementation of the neural network framework is complecated. Although I have not implement backward propagation, I learned somthing about how to accelerate GEMM, which i I think it is the core of DNN. 
