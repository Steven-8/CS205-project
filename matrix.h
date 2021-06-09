//
// Created by 23532 on 2021/6/6.
//
#include<vector>

#ifndef CS205_PROJECT_MATRIX_H
#define CS205_PROJECT_MATRIX_H
using namespace std;

template<typename T>
class Matrix {
protected:
    vector<vector<T>> matrix;
public:
    Matrix();

    Matrix(vector<vector<T>> mat);

    int get_rows();

    int get_cols();

    vector<vector<T>> &getMatrix();

    virtual void transpose();

    friend ostream &operator<<(ostream &os, Matrix<T> mat) {
        for (int i = 0; i < mat.matrix.size(); ++i) {
            for (int j = 0; j < mat.matrix.front().size(); ++j) {
                os << mat.matrix[i][j];
                os << " ";
            }
            os << endl;
        }
        return os;
    }
};

template<typename T>
class sparseMatrix : public Matrix<T> {
private:
    vector<T> value;
    vector<int[2]> position;
public:
    sparseMatrix(vector<vector<T>> mat);
    void transpose();
};

//加法
template<typename T>
Matrix<T> operator+(Matrix<T> mat1, Matrix<T> mat2);

//减法
template<typename T>
Matrix<T> operator-(Matrix<T> mat1, Matrix<T> mat2);

//标量乘法
template<typename T>
Matrix<T> scalar_mul(Matrix<T> mat, T t);

//标量除法
template<typename T>
Matrix<T> scalar_div(Matrix<T> mat, T t);

//共轭矩阵
template<typename T>
Matrix<T> conjugation(Matrix<T> mat);

//矩阵元素相乘
template<typename T>
Matrix<T> e_w_mul(Matrix<T> mat1, Matrix<T> mat2);

//矩阵相乘
template<typename T>
Matrix<T> operator*(Matrix<T> mat1, Matrix<T> mat2);

//矩阵向量相乘
template<typename T>
Matrix<T> operator*(Matrix<T> mat, vector<T> vec);

//乘法
template<typename T>
Matrix<T> operator*(vector<T> vec, Matrix<T> mat);

//点乘
template<typename T>
T dot(vector<T> vec1, vector<T> vec2);

//叉乘
template<typename T>
vector<T> cross(vector<T> vec1, vector<T> vec2);

template<typename T>
T max_val(Matrix<T> mat);

template<typename T>
T max_row_val(Matrix<T> mat, int row_index);

template<typename T>
T max_col_val(Matrix<T> mat, int col_index);

template<typename T>
T min_val(Matrix<T> mat);

template<typename T>
T min_row_val(Matrix<T> mat, int row_index);

template<typename T>
T min_col_val(Matrix<T> mat, int col_index);

template<typename T>
T sum(Matrix<T> mat);

template<typename T>
T sum_row(Matrix<T> mat, int row_index);

template<typename T>
T sum_col(Matrix<T> mat, int col_index);

template<typename T>
T avg(Matrix<T> mat);

template<typename T>
T avg_row(Matrix<T> mat, int row_index);

template<typename T>
T avg_col(Matrix<T> mat, int col_index);


//卷积
template<typename T>
Matrix<T> convolution(Matrix<T> kernel, Matrix<T> mat);

#endif //CS205_PROJECT_MATRIX_H
