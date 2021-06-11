//
// Created by wendy on 2021/6/10.
//
#include "relu.h"
#include<omp.h>
void relu(Data &input){
    // x = max(x,0)
    int size = input.batch * input.depth * input.height * input.width;
#pragma omp parallel for
    for(int i = 0; i < size; i++){
        input.data[i] = input.data[i] > 0 ? input.data[i] : 0;
    }
}