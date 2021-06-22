//
// Created by wendy on 2021/6/9.
//
#include "data.h"
#include<iostream>

using namespace std;
Data::Data(){
    data = nullptr;
    batch = depth = height = width = 0;
}
Data::Data(int n,int c, int h, int w){
    batch = n; depth = c; height = h; width = w;
    int size = n * c * h * w;
    data = new float[size];
};
Data::~Data(){
    if(data)
        delete [] data;
}

Data &Data::operator = (const Data & chunks){
    batch = chunks.batch;
    depth = chunks.depth;
    height = chunks.height;
    width = chunks.width;
    if(data)
        delete [] data;
    data = new float[batch * depth * height * width];
    for (int i = 0; i < batch * depth * width * height; i++)
        data[i] = chunks.data[i];
    return *this;
}
void Data::setData(float * input){
   int size = getSize();
   if(data) {
       for (int i = 0; i < size; i++) {
           data[i] = input[i];
       }
   }
}
void Data::resetValue(int n,int c, int h, int w){
    batch = n; depth = c; height = h; width = w;
    if(data){
        delete [] data;
    }
    int size = n * c * h * w;
    data = new float[size];
}
int Data::getSize(){
    return batch * depth * height * width;
}
float Data::getValue(int i, int j, int m, int n){
    int inx = getIndex(i,j,m,n);
    return data[inx];
}
int Data::getIndex(int i, int j, int m, int n){
    int inx = i * ( depth * width * height) + j * ( width * height) + m * height + n;
    return inx;
}
void Data::SetValue(int i, int j, int m, int n,float  value){
    int inx = i * ( depth * width * height) + j * ( width * height) + m * height + n;
    data[inx] = value;
}
void Data::AddValue(int i, int j, int m, int n, float value){
    int inx = i * ( depth * width * height) + j * ( width * height) + m * height + n;
    data[inx] += value;
}
void Data::RandomInit(){
    int N = 999;
    srand(time(NULL)); // 随机种子
    for (int i = 0; i < batch * depth * width * height; i++)
        data[i] = rand() % (N + 1) / (float)(N + 1);
}
void Data::Init(float fillValue){
    for (int i = 0; i < batch * depth * width * height; i++)
        data[i] = fillValue;
}
void Data::print(){
    for (int i = 0; i < batch * depth * width * height; i++){
        cout << data[i] << " ";
        if((i+1) % (depth * width * height) == 0){
            cout << endl;
        }
    }

}
// general functions for computing output shape
int computeShape(int inputLength, int  padding, int stride, int kernel){
    int output = (inputLength + 2 * padding - kernel) / stride + 1;
    return output;
}
