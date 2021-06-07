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

template<typename T>
T max_val(Matrix<T> mat);
template<typename T>
T max_row_val(Matrix<T> mat, int row_index);
template<typename T>
T max_col_val(Matrix<T> mat,int col_index);
template<typename T>
T min_val(Matrix<T> mat);
template<typename T>
T min_row_val(Matrix<T> mat,int row_index);
template<typename T>
T min_col_val(Matrix<T> mat,int col_index);
template<typename T>
T sum(Matrix<T> mat);
template<typename T>
T sum_row(Matrix<T> mat,int row_index);
template<typename T>
T sum_col(Matrix<T> mat,int col_index);
template<typename T>
T avg(Matrix<T> mat);
template<typename T>
T avg_row(Matrix<T> mat,int row_index);
template<typename T>
T avg_col(Matrix<T> mat,int col_index);

#endif //CS205_PROJECT_MATRIX_H
