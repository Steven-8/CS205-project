#include <iostream>
#include <vector>
#include<complex>
#include "matrix.h"
using namespace std;
//大小不匹配异常
class size_mismatch_exception : public exception {
public:
    const char *what() {
        return "matrix size mismatch ";
    }
};

//类型不匹配异常
class type_mismatch_exception : public exception {
public:
    const char *what() {
        return "matrix type mismatch";
    }
};




template<typename T>
Matrix<T>::Matrix(vector<vector<T>> mat) {
    this->matrix = mat;
}

template<typename T>
int Matrix<T>::get_rows() {
    return this->matrix.size();
}

template<typename T>
int Matrix<T>::get_cols() {
    return this->matrix.front().size();
}

template<typename T>
vector<vector<T>>& Matrix<T>::getMatrix() {
    return matrix;
}

//矩阵转置
template<typename T>
void Matrix<T>::transpose() {
    vector<vector<T>> vec;
    for (int i = 0; i < get_cols(); ++i) {
        vector<T> temp;
        for (int j = 0; j < get_rows(); ++j) {
            temp.push_back(matrix[j][i]);
        }
        vec.push_back(temp);
    }
    matrix = vec;
    return;
};

//加法
template<typename T>
Matrix<T> operator+(Matrix<T> mat1, Matrix<T> mat2) {
    vector<vector<T>> vec;
    if (mat1.get_rows() != mat2.get_rows() || mat1.get_cols() != mat2.get_cols()) {
        throw size_mismatch_exception();
    } else {
        for (int i = 0; i < mat1.get_rows(); ++i) {
            vector<T> temp;
            for (int j = 0; j < mat1.get_cols(); ++j) {
                temp.push_back(mat1.getMatrix()[i][j] + mat2.getMatrix()[i][j]);
            }
            vec.push_back(temp);
        }
    }
    return Matrix<T>(vec);
}

//减法
template<typename T>
Matrix<T> operator-(Matrix<T> mat1, Matrix<T> mat2) {
    vector<vector<T>> vec;
    if (mat1.get_rows() != mat2.get_rows() || mat1.get_cols() != mat2.get_cols()) {
        throw size_mismatch_exception();
    } else {
        for (int i = 0; i < mat1.get_rows(); ++i) {
            vector<T> temp;
            for (int j = 0; j < mat1.get_cols(); ++j) {
                temp.push_back(mat1.getMatrix()[i][j] - mat2.getMatrix()[i][j]);
            }
            vec.push_back(temp);
        }
    }
    return Matrix<T>(vec);
}

//标量乘法
template<typename T>
Matrix<T> scalar_mul(Matrix<T> mat, T t) {
    vector<vector<T>> vec;
    for (int i = 0; i < mat.get_rows(); ++i) {
        vector<T> temp;
        for (int j = 0; j < mat.get_cols(); ++j) {
            temp.push_back(mat.getMatrix()[i][j] * t);
        }
        vec.push_back(temp);
    }
    return Matrix<T>(vec);
}

//标量除法
template<typename T>
Matrix<T> scalar_div(Matrix<T> mat, T t) {
    vector<vector<T>> vec;
    for (int i = 0; i < mat.get_rows(); ++i) {
        vector<T> temp;
        for (int j = 0; j < mat.get_cols(); ++j) {
            temp.push_back(mat.getMatrix()[i][j] / t);
        }
        vec.push_back(temp);
    }
    return Matrix<T>(vec);
}

//置换矩阵 ?返回多个矩阵
template<typename T>
Matrix<T> transposition(Matrix<T> &mat) {

}


//共轭矩阵 ?如何判断是否为复数
/*
template<typename T>
Matrix<T> conjugation(Matrix<T> &mat) {
    mat.transpose();
    vector<vector<T>> vec;
    if (typeid(T) == typeid(complex<int>) || typeid(T) == typeid(complex<double>)) {
        for (int i = 0; i < mat.get_rows(); ++i) {
            vector<T> temp;
            for (int j = 0; j < mat.get_cols(); ++j) {
                mat.getMatrix()[i][j].imag() = -mat.getMatrix()[i][j].imag;
                temp.push_back();
            }
        }
    }
}

*/

//矩阵元素相乘
template<typename T>
Matrix<T> e_w_mul(Matrix<T> mat1, Matrix<T> mat2) {
    if (mat1.get_rows() != mat2.get_rows() || mat1.get_cols() != mat2.get_cols()) {
        throw size_mismatch_exception();
    }
    vector<vector<T>> vec;
    for (int i = 0; i < mat1.get_rows(); ++i) {
        vector<T> temp;
        for (int j = 0; j < mat1.get_cols(); ++j) {
            temp.push_back(mat1.getMatrix()[i][j] * mat2.getMatrix()[i][j]);
        }
        vec.push_back(temp);
    }

    return Matrix<T>(vec);
}

//矩阵相乘
template<typename T>
Matrix<T> operator*(Matrix<T> mat1, Matrix<T> mat2) {
    if (mat1.get_cols() != mat2.get_rows()) {
        throw size_mismatch_exception();
    }
    vector<vector<T>> vec;
    for (int i = 0; i < mat1.get_rows(); ++i) {
        vector<T> temp;
        for (int j = 0; j < mat2.get_cols(); ++j) {
            T sum = 0;
            for (int k = 0; k < mat1.get_cols(); ++k) {
                sum += mat1.getMatrix()[i][k] * mat2.getMatrix()[k][j];
            }
            temp.push_back(sum);
        }
        vec.push_back(temp);
    }

    return Matrix<T>(vec);
}

//矩阵向量相乘
template<typename T>
Matrix<T> operator*(Matrix<T> mat, vector<T> vec) {
    if (mat.get_cols() != vec.size()) {
        throw size_mismatch_exception();
    }
    vector<vector<T>> new_vec;
    for (int i = 0; i < mat.get_rows(); ++i) {
        vector<T> temp;
        T sum = 0;
        for (int j = 0; j < mat.get_cols(); ++j) {
            sum += mat.getMatrix()[i][j] * vec[j];
        }
        temp.push_back(sum);
        new_vec.push_back(temp);
    }

    return Matrix<T>(new_vec);
}

template<typename T>
Matrix<T> operator*(vector<T> vec, Matrix<T> mat) {
    if (mat.get_rows() != vec.size()) {
        throw size_mismatch_exception();
    }
    vector<vector<T>> new_vec;
    vector<T> temp;
    for (int i = 0; i < mat.get_cols(); ++i) {
        T sum = 0;
        for (int j = 0; j < mat.get_rows(); ++j) {
            sum += mat.getMatrix()[j][i] * vec[j];
        }
        temp.push_back(sum);
    }
    new_vec.push_back(temp);
    return Matrix<T>(new_vec);
}

//点乘
template<typename T>
T dot(vector<T> vec1, vector<T> vec2) {
    if (vec1.size() != vec2.size()) {
        throw size_mismatch_exception();
    }
    T sum = 0;
    for (int i = 0; i < vec1.size(); ++i) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

//叉乘
template<typename T>
vector<T> cross(vector<T> vec1, vector<T> vec2) {
    if (vec1.size() != 3 || vec2.size() != 3) {
        throw size_mismatch_exception();
    }
    vector<T> vec;
    vec.push_back(vec1[1] * vec2[2] - vec1[2] * vec2[1]);
    vec.push_back(vec1[0] * vec2[2] - vec1[2] * vec2[0]);
    vec.push_back(vec1[0] * vec2[1] - vec1[1] * vec2[0]);
    return vec;
}

int main(){
    vector<int > r1;
    vector<int > r2;
    vector<int > r3;
    r1.push_back(1);
    r1.push_back(2);
    r1.push_back(3);
    r2.push_back(4);
    r2.push_back(5);
    r2.push_back(6);
    r3.push_back(7);
    r3.push_back(8);
    r3.push_back(9);
    vector<vector<int>> mat;
    mat.push_back(r1);
    mat.push_back(r2);
    mat.push_back(r3);

    Matrix<int> m1(mat);
    Matrix<int> m2=m1+m1+m1;
    m2= scalar_mul(m2,3);
    for(int i=0; i<m2.get_rows();i++){
        for (int j = 0; j < m2.get_cols(); ++j) {
            cout<<m2.getMatrix()[i][j]<<"  ";
        }
        cout<<endl;

    }





}
