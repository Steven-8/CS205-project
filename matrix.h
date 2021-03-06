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

    Matrix(T** mat,int rows,int cols);
    Matrix(vector<vector<T>> mat);

    ~Matrix(){};
    int get_rows();

    int get_cols();

    vector<vector<T>> &getMatrix();
    Matrix<T>getTranspose();
    Matrix<double>getInverse();
    double determinant();
    void reshape(int x1,int y1,int x2,int y2);
    void row_slicing(int row_index);
    void col_slicing(int col_index);

    virtual void transpose();

    friend ostream &operator<<(ostream &os, Matrix<T> mat) {
        for (int i = 0; i < mat.get_rows(); ++i) {
            for (int j = 0; j < mat.get_cols(); ++j) {
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

//5
template<typename T>
double arrayMultiplyAndAdd(T* first,T* second, int len);
template<typename  T>
Matrix<double>GramSchimidt(Matrix<T> paraArray);
template<typename  T>
Matrix<double> QRdecomposition(Matrix<T> paraMatrix);
template<typename T>
Matrix<double> Eigenvalue(Matrix<T> paraMatrix, int paraIter);
Matrix<double> arrayRowValue(Matrix<double> tempSummary,int* paraIndexQ,int size);
int* arrayIndexAuto(int start, int end);
template<typename T>
Matrix<double> Eigenvector(Matrix<double> paraMatrix, int paraIter);
template<typename T>
T trace(Matrix<T> mat);
#endif //CS205_PROJECT_MATRIX_H
