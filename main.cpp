#include <iostream>
#include <vector>

using namespace std;

class mismatch_exception : public exception {
public:
    const char *what() {
        return "matrix mismatch exception";
    }
};

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

template<typename T>
class Matrix {
private:
    vector<vector<T>> matrix;
public:
    Matrix(vector<vector<T>> &mat);

    int get_rows();

    int get_cols();

    Matrix<T> transpose();
    Matrix<T> operator+(Matrix<T> &mat);
    Matrix<T> operator-(Matrix<T> &mat);


};

template<typename T>
Matrix<T>::Matrix(vector<vector<T>> &mat) {
    this->matrix = *mat;
}

template<typename T>
int Matrix<T>::get_rows() {
    return this->matrix.size();
}

template<typename T>
int Matrix<T>::get_cols() {
    return this->matrix.front().size();
}

//矩阵转置
template<typename T>
Matrix<T> Matrix<T>::transpose() {
    vector<vector<T>> vec;
    for (int i = 0; i < get_cols(); ++i) {
        vector<T> temp;
        for (int j = 0; j < get_rows(); ++j) {
            temp.push_back(matrix[j][i]);
        }
        vec.push_back(temp);
    }
    matrix = vec;
    return *this;
};

//加法
template<typename T>
Matrix<T> Matrix<T>::operator+(Matrix<T> &mat) {
    vector<vector<T>> vec;
    if (mat.get_rows() != this->get_rows() || mat.get_cols() != this->get_cols()) {
        throw mismatch_exception();
    } else {
        for (int i = 0; i < mat.get_rows(); ++i) {
            vector<T> temp;
            for (int j = 0; j < mat.get_cols(); ++j) {
                temp.push_back(this->matrix[i][j] + mat[i][j]);
            }
            vec.push_back(temp);
        }
    }
    return Matrix(&vec);
}

//减法
template<typename T>
Matrix<T> Matrix<T>::operator-(Matrix<T> &mat) {
    vector<vector<T>> vec;
    if (mat.get_rows() != this->get_rows() || mat.get_cols() != this->get_cols()) {
        throw mismatch_exception();
    } else {
        for (int i = 0; i < mat.get_rows(); ++i) {
            vector<T> temp;
            for (int j = 0; j < mat.get_cols(); ++j) {
                temp.push_back(this->matrix[i][j] - mat[i][j]);
            }
            vec.push_back(temp);
        }
    }
    return Matrix(&vec);
}

//标量乘法
template<typename T>
Matrix<T> scalar_mul(Matrix<T> &mat, T t) {
    for (int i = 0; i < mat.get_rows(); ++i) {
        for (int j = 0; j < mat.get_cols(); ++j) {
            mat[i][j] = mat[i][j] * t;
        }
    }
    return mat;
}

//标量除法
template<typename T>
Matrix<T> scalar_div(Matrix<T> &mat, T t) {
    for (int i = 0; i < mat.get_rows(); ++i) {
        for (int j = 0; j < mat.get_cols(); ++j) {
            mat[i][j] = mat[i][j] / t;
        }
    }
    return mat;
}

//共轭矩阵
