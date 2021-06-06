//
// Created by 23532 on 2021/6/6.
//
#include<vector>
#ifndef CS205_PROJECT_MATRIX_H
#define CS205_PROJECT_MATRIX_H
using namespace std;
template<typename T>
class Matrix {
private:
    vector<vector<T>> matrix;
public:
    Matrix(vector<vector<T>> mat);

    int get_rows();

    int get_cols();

    vector<vector<T>>& getMatrix();

    void transpose();

};
#endif //CS205_PROJECT_MATRIX_H
