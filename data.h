//
// Created by wendy on 2021/6/9.
//

#ifndef DATA_H
#define DATA_H

class Data{
public:
    int batch;
    int depth;
    int width;
    int height;
    float *data;
    // function
    Data();
    Data(int n,int c, int h, int w);
    ~Data();
    // copy
    Data& operator=(const Data & chunks);
    float getValue(int i, int j, int m, int n);
    int getIndex(int i, int j, int m, int n);
    void SetValue(int i, int j, int m, int n, float  value);
    void AddValue(int i, int j, int m, int n, float value);
    void resetValue(int n,int c, int h, int w);
    void setData(float * input);
    int getSize();
    void RandomInit();
    void Init(float fillValue = 0);
    void print();
};

int computeShape(int inputLength,int  padding, int stride, int kernel);
#endif// DATA_H
