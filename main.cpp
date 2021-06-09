#include <iostream>
#include <vector>
#include<complex>
#include "matrix.h"
#include <typeinfo>
#include <string>
#include <cmath>
using namespace std;

//大小不匹配异常
class size_mismatch_exception : public exception {
public:
    size_mismatch_exception(string str) {
        cout << str;
    }
};

//类型不匹配异常
class type_mismatch_exception : public exception {
public:
    type_mismatch_exception(string str) {
        cout << str;
    }
};

template<typename T>
Matrix<T>::Matrix() {
}

template<typename T>
Matrix<T>::Matrix(T** mat,int rows,int cols) {
    vector<vector<T>> vec;
    for(int i=0;i<rows;i++){
        vector<T> t;
        for(int j=0;j<cols;j++){
            t.push_back(mat[i][j]);
        }
        vec.push_back(t);
    }
    this->matrix=vec;
}
template<typename T>
Matrix<T>::Matrix(vector<vector<T>> mat) {
    try {
        for (int i = 0; i < mat.size(); ++i) {
            if (i == mat.size() - 1) {
                break;
            }
            if (mat[i].size() != mat[i + 1].size()) {
                throw size_mismatch_exception("the matrix is invalid");
            }
        }
    } catch (size_mismatch_exception e) {
        exit(-1);
    }
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
vector<vector<T>> &Matrix<T>::getMatrix() {
    return matrix;
}

//正常矩阵转置
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
template<typename T>
Matrix<T> Matrix<T>::getTranspose(){
    Matrix<T> mat(this->getMatrix());
    mat.transpose();
    return mat;
};
template<typename T>
Matrix<T>inverse(){

};


//稀疏矩阵构造函数
template<typename T>
sparseMatrix<T>::sparseMatrix(vector<vector<T>> mat):Matrix<T>(mat) {
    for (int i = 0; i < mat.size(); ++i) {
        for (int j = 0; j < mat.front().size(); ++j) {
            if (mat[i][j] != 0) {
                position.push_back({i,j});
                value.push_back(mat[i][j]);
            }
        }
    }
}

//稀疏矩阵转置
template<typename T>
void sparseMatrix<T>::transpose() {
    for (int i = 0; i < value.size(); ++i) {
        int x = position[i][0];
        int y = position[i][0];
        position[i][0] = x;
        position[i][1] = y;
        T temp = this->matrix[x][y];
        this->matrix[x][y] = this->matrix[y][x];
        this->matrix[y][x] = temp;
    }
}

//加法
template<typename T>
Matrix<T> operator+(Matrix<T> mat1, Matrix<T> mat2) {
    vector<vector<T>> vec;
    if (mat1.get_rows() != mat2.get_rows() || mat1.get_cols() != mat2.get_cols()) {
        try {
            throw size_mismatch_exception("two matrix must have the same size");
        } catch (size_mismatch_exception e) {
            exit(-1);
        }
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
        try {
            throw size_mismatch_exception("two matrix must have the same size");
        }
        catch (size_mismatch_exception e) {
            exit(-1);
        }
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


//共轭矩阵
template<typename T>
Matrix<T> conjugation(Matrix<T> mat) {
    mat.transpose();
    vector<vector<T>> vec;
    string s1(typeid(T).name());
    s1 = s1.substr(0, 10);
    string s2(typeid(complex<double>).name());
    s2 = s2.substr(0, 10);
    if (s1 != s2) {
        try {
            throw type_mismatch_exception("the element of the matrix must be complex");
        }
        catch (type_mismatch_exception e) {
            exit(-1);
        }
    } else {
        for (int i = 0; i < mat.get_rows(); ++i) {
            vector<T> temp;
            for (int j = 0; j < mat.get_cols(); ++j) {
                temp.push_back(conj(mat.getMatrix()[i][j]));
            }
            vec.push_back(temp);
        }
    }
    return Matrix<T>(vec);
}


//矩阵元素相乘
template<typename T>
Matrix<T> e_w_mul(Matrix<T> mat1, Matrix<T> mat2) {
    try {
        if (mat1.get_rows() != mat2.get_rows() || mat1.get_cols() != mat2.get_cols()) {
            throw size_mismatch_exception("two matrix's size must be the same");
        }
    }
    catch (size_mismatch_exception e) {
        exit(-1);
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
    try {
        if (mat1.get_cols() != mat2.get_rows()) {
            throw size_mismatch_exception("matrix1's col must be equal to matrix2's row ");
        }
    } catch (size_mismatch_exception e) {
        exit(-1);
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
    try {
        if (mat.get_rows() != vec.size()) {
            throw size_mismatch_exception("vector's size must be equal to matrix's row");
        }
    }
    catch (size_mismatch_exception e) {
        exit(-1);
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

//乘法
template<typename T>
Matrix<T> operator*(vector<T> vec, Matrix<T> mat) {
    try {
        if (mat.get_rows() != vec.size()) {
            throw size_mismatch_exception("vector's size must be equal to matrix's row");
        }
    }
    catch (size_mismatch_exception e) {
        exit(-1);
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
    try {
        if (vec1.size() != vec2.size()) {
            throw size_mismatch_exception("two vector must have the same size");
        }
    } catch (size_mismatch_exception e) {
        exit(-1);
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
    try {
        if (vec1.size() != 3 || vec2.size() != 3) {
            throw size_mismatch_exception("vector's size for cross is not valid");
        }
    } catch (size_mismatch_exception e) {
        exit(-1);
    }
    vector<T> vec;
    vec.push_back(vec1[1] * vec2[2] - vec1[2] * vec2[1]);
    vec.push_back(vec1[0] * vec2[2] - vec1[2] * vec2[0]);
    vec.push_back(vec1[0] * vec2[1] - vec1[1] * vec2[0]);
    return vec;
}

//part4
template<typename T>
T max_val(Matrix<T> mat) {
    T t = mat.getMatrix()[0][0];
    for (int i = 0; i < mat.get_rows(); i++) {
        for (int j = 0; j < mat.get_cols(); j++) {
            if (t < mat.getMatrix()[i][j]) {
                t = mat.getMatrix()[i][j];
            }
        }

    }
    return t;
};

template<typename T>
T max_row_val(Matrix<T> mat, int row_index) {
    try {
        if (mat.get_rows() < row_index) {
            throw row_index;
        }
    } catch (int row_index) {
        cout << "exception: row_index out of bound: " << row_index << endl;
        exit(-1);
    }
    T t = mat.getMatrix()[0][0];
    for (int i = 0; i < mat.get_cols(); ++i) {
        if (t < mat.getMatrix()[row_index - 1][i]) {
            t = mat.getMatrix()[row_index - 1][i];
        }
    }
    t = t - mat.getMatrix()[0][0];
    return t;

};

template<typename T>
T max_col_val(Matrix<T> mat, int col_index) {
    try {
        if (mat.get_cols() < col_index) {
            throw col_index;
        }
    } catch (int col_index) {
        cout << "exception: col_index out of bound: " << col_index << endl;
        exit(-1);
    }
    T t = mat.getMatrix()[0][0];
    for (int i = 0; i < mat.get_rows(); ++i) {
        if (t < mat.getMatrix()[i][col_index - 1]) {
            t = mat.getMatrix()[i][col_index - 1];
        }
    }
    t = t - mat.getMatrix()[0][0];
    return t;

};;

template<typename T>
T min_val(Matrix<T> mat) {
    T t = mat.getMatrix()[0][0];
    for (int i = 0; i < mat.get_rows(); i++) {
        for (int j = 0; j < mat.get_cols(); j++) {
            if (t > mat.getMatrix()[i][j]) {
                t = mat.getMatrix()[i][j];
            }
        }

    }
    return t;
};

template<typename T>
T min_row_val(Matrix<T> mat, int row_index) {
    try {
        if (mat.get_rows() < row_index) {
            throw row_index;
        }
    } catch (int row_index) {
        cout << "exception: row_index out of bound: " << row_index << endl;
        exit(-1);
        exit(-1);
    }

    T t = mat.getMatrix()[0][0];
    for (int i = 0; i < mat.get_cols(); ++i) {
        if (t > mat.getMatrix()[row_index - 1][i]) {
            t = mat.getMatrix()[row_index - 1][i];
        }
    }
    t = t - mat.getMatrix()[0][0];
    return t;
};

template<typename T>
T min_col_val(Matrix<T> mat, int col_index) {
    try {
        if (mat.get_cols() < col_index) {
            throw col_index;
        }
    } catch (int col_index) {
        cout << "exception: col_index out of bound: " << col_index << endl;
        exit(-1);

    }
    T t = mat.getMatrix()[0][0];
    for (int i = 0; i < mat.get_rows(); ++i) {
        if (t > mat.getMatrix()[i][col_index - 1]) {
            t = mat.getMatrix()[i][col_index - 1];
        }
    }
    t = t - mat.getMatrix()[0][0];
    return t;

};

template<typename T>
T sum(Matrix<T> mat) {
    T t = mat.getMatrix()[0][0];
    for (int i = 0; i < mat.get_rows(); i++) {
        for (int j = 0; j < mat.get_cols(); j++) {
            t = t + mat.getMatrix()[i][j];
        }
    }
    t = t - mat.getMatrix()[0][0];
    return t;

};

template<typename T>
T sum_row(Matrix<T> mat, int row_index) {
    try {
        if (mat.get_rows() < row_index) {
            throw row_index;
        }
    } catch (int row_index) {
        cout << "exception: row_index out of bound: " << row_index << endl;
        exit(-1);
    }
    T t = mat.getMatrix()[0][0];
    for (int i = 0; i < mat.get_cols(); ++i) {
        t = t + mat.getMatrix()[row_index - 1][i];
    }
    t = t - mat.getMatrix()[0][0];
    return t;

};

template<typename T>
T sum_col(Matrix<T> mat, int col_index) {
    try {
        if (mat.get_cols() < col_index) {
            throw col_index;
        }
    } catch (int col_index) {
        cout << "exception: col_index out of bound: " << col_index << endl;
        exit(-1);
    }
    T t = mat.getMatrix()[0][0];
    for (int i = 0; i < mat.get_rows(); i++) {
        t = t + mat.getMatrix()[i][col_index - 1];
    }
    t = t - mat.getMatrix()[0][0];
    return t;
};

template<typename T>
T avg(Matrix<T> mat) {
    return sum(mat) / (mat.get_rows() * mat.get_cols());
};

template<typename T>
T avg_row(Matrix<T> mat, int row_index) {
    return sum_row(mat, row_index) / mat.get_cols();
};

template<typename T>
T avg_col(Matrix<T> mat, int col_index) {
    return sum_col(mat, col_index) / mat.get_rows();
};


//卷积
template<typename T>
Matrix<T> convolution(Matrix<T> kernel, Matrix<T> mat) {
    try {
        if (kernel.get_cols() != kernel.get_rows() || kernel.get_rows() % 2 == 0) {
            throw size_mismatch_exception("kernel's size is invalid");
        }
    }
    catch (size_mismatch_exception e) {
        exit(-1);
    }
    int n = kernel.get_rows() / 2;
    vector<vector<T>> new_mat;
    for (int i = 0; i < 2 * n + mat.get_rows(); ++i) {
        vector<T> temp;
        for (int j = 0; j < 2 * n + mat.get_cols(); ++j) {
            if (i >= n && i < n + mat.get_rows()) {
                if (j < n || j >= n + mat.get_cols()) {
                    temp.push_back(0);
                } else {
                    temp.push_back(mat.getMatrix()[i - n][j - n]);
                }
            } else {
                temp.push_back(0);
            }
        }
        new_mat.push_back(temp);
    }
    cout << Matrix<T>(new_mat);
    vector<vector<T>> vec;
    for (int i = 0; i < mat.get_rows(); ++i) {
        vector<T> temp;
        for (int j = 0; j < mat.get_cols(); ++j) {
            int sum = 0;
            for (int k = 0; k < kernel.get_rows(); ++k) {
                for (int l = 0; l < kernel.get_cols(); ++l) {
                    sum += kernel.getMatrix()[k][l] * new_mat[k + i][j + l];
                }
            }
            temp.push_back(sum);
        }
        vec.push_back(temp);
    }
    return vec;
}


//5
template<typename T>
double arrayMultiplyAndAdd(T* first,T* second, int len){
    double result=0.0;
    for(int i=0;i<len;i++){
        result+=(double)(first[i]*second[i]);
    }
    return result;
};
template<typename  T>
Matrix<double>GramSchimidt(Matrix<T> paraArray){
    Matrix<T> tempTransposeMatrix=paraArray.getTranspose();
    int m=tempTransposeMatrix.get_rows();
    int n=tempTransposeMatrix.get_cols();

    double** tempmatrix=new double*[m];
    for(int i=0;i<m;i++){
        tempmatrix[i]=new double[n];
    }

    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            tempmatrix[i][j]=tempTransposeMatrix.getMatrix()[i][j];
        }
    }

    double** resultMatrix=new double*[m];
    for(int i=0;i<m;i++){
        resultMatrix[i]=new double[n];
    };
    double tempVal=0,tempFactor=0;
    for (int i = 0; i < m; ++i) {
        for(int j=0;j<n;j++){
            tempVal=tempmatrix[i][j];
            for (int k = 0; k < i; ++k) {
                tempFactor=(1.0* arrayMultiplyAndAdd(tempmatrix[i],resultMatrix[k],n))/ arrayMultiplyAndAdd(resultMatrix[k],resultMatrix[k],n);
                tempVal-=tempFactor*resultMatrix[k][j];
            }
            resultMatrix[i][j]=tempVal;
        }

    }
    Matrix<double> mat(resultMatrix,m,n);
    return mat.getTranspose();

};

template<typename  T>
Matrix<double> QRdecomposition(Matrix<T> paraMatrix){
    Matrix<double> tempOrthogonalMatrix = GramSchimidt(paraMatrix).getTranspose();
    int m=tempOrthogonalMatrix.get_rows();
    int n=tempOrthogonalMatrix.get_cols();

    double** tempmatrix=new double*[m];
    for(int i=0;i<m;i++){
        tempmatrix[i]=new double[n];
    }
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            tempmatrix[i][j]=tempOrthogonalMatrix.getMatrix()[i][j];
        }
    }

    vector<vector<double>> vec;
    for(int i=0;i<m;i++){
        double tempMag = sqrt(arrayMultiplyAndAdd(tempmatrix[i],tempmatrix[i],n));
        vector<double> v;
        for(int j=0;j<n;j++){
            v.push_back(tempmatrix[i][j] / tempMag);
        }
        vec.push_back(v);
    }
    Matrix<double> tempMatrixQ(vec);
    int tempM=paraMatrix.get_rows();
    int tempN=paraMatrix.get_cols();
    double** temppara=new double*[tempM];
    for(int i=0;i<m;i++){
        temppara[i]=new double[tempN];
    }
    for(int i=0;i<tempM;i++){
        for(int j=0;j<tempN;j++){
            temppara[i][j]=paraMatrix.getMatrix()[i][j];
        }
    }
    Matrix<double> tempParaMatrix(temppara,tempM,tempN);
    Matrix<double> tempMatrixR = tempMatrixQ * tempParaMatrix;


    //
    //cout<<tempMatrixR<<endl;
    //delete[] temppara;
    //tempParaMatrix.~Matrix();
    //
    double** resultSummary=new double*[m+n];
    for(int i=0;i<m+n;i++){
        resultSummary[i]=new double[m];
    }
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            resultSummary[i][j]=tempMatrixQ.getMatrix()[j][i];
        }
    }
    for(int i=n;i<n+m;i++){
        for(int j=0;j<m;j++){
            resultSummary[i][j]=tempMatrixR.getMatrix()[i-n][j];
        }
    }
    Matrix<double> result(resultSummary,n+m,m);
    return result;
};

int* arrayIndexAuto(int start, int end){
    int* pointer=new int[end-start];
    for (int i = 0; i < end-start; i++) {
        pointer[i] = i + start;
    }
    return pointer;
};
Matrix<double> arrayRowValue(Matrix<double> tempSummary,int* paraIndexQ,int size){
    vector<vector<double>> vec;
    for(int i=0;i<size;i++){
        vector<double> t=tempSummary.getMatrix()[paraIndexQ[i]];
        vec.push_back(t);
    }
    return Matrix<double>(vec);
};
template<typename T>
Matrix<double> Eigenvalue(Matrix<T> paraMatrix, int paraIter){
    int m=paraMatrix.get_rows();
    int n=paraMatrix.get_cols();
    int* tempIndexQ= arrayIndexAuto(0,m);
    int* tempIndexR = arrayIndexAuto(m,n+m);
    Matrix<double> result;
    for(int i=0;i<paraIter;i++){
        Matrix<double> tempSummary= QRdecomposition(paraMatrix);
        Matrix<double> tempMatrixQ= arrayRowValue(tempSummary,tempIndexQ,m);
        Matrix<double> tempMatrixR= arrayRowValue(tempSummary,tempIndexR,n);
        result=tempMatrixR*tempMatrixQ;
    }

    return result;
};
template<typename T>
Matrix<double> Eigenvector(Matrix<T> paraMatrix, int paraIter){
    int m=paraMatrix.get_rows();
    int n=paraMatrix.get_cols();
    Matrix<double> tempMatrix= Eigenvalue(paraMatrix,paraIter);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(i!=j){
                tempMatrix.getMatrix()[i][j]=0;
            }
        }
    }
    cout<<tempMatrix;
    tempMatrix.getMatrix()[0][0]=0;
    cout<<tempMatrix;
    return (paraMatrix-tempMatrix).inverse();
};
int main() {
    vector<int> r1;
    vector<int> r2;
    vector<int> r3;
    vector<int> r4;
    vector<int> r5;
    r1.push_back(1);
    r1.push_back(2);
    r1.push_back(3);
//    r1.push_back(3);
//    r1.push_back(3);
    r2.push_back(4);
    r2.push_back(5);
    r2.push_back(6);
//    r2.push_back(6);
    r3.push_back(7);
    r3.push_back(8);
    r3.push_back(9);
//    r3.push_back(9);
 //   r3.push_back(9);
    /*r4.push_back(1);
    r4.push_back(2);
    r4.push_back(3);
    r4.push_back(3);
    r4.push_back(3);
    r5.push_back(4);
    r5.push_back(5);
    r5.push_back(6);
    r5.push_back(6);
    r5.push_back(6);*/
    vector<vector<int>> mat;
    mat.push_back(r1);
    mat.push_back(r2);
    mat.push_back(r3);
//    mat.push_back(r4);
//    mat.push_back(r5);
    Matrix<int> m1(mat);
    Matrix<int> m2(m1);
    Matrix<double> m4=Eigenvalue(m2,10);
    cout<<m4;
    Eigenvector(m2,1);

/*    sparseMatrix<int> m3(mat);
    cout<<m3;
    m3.transpose();
    cout<<m3;*/
    //  cout << m1 << endl;
//    m1.transpose();
//    cout<<m1<<endl;
//    cout<<m1+m2<<endl;
//    cout<<m1-m2<<endl;
//    cout<<m1*m2<<endl;
//    cout<<e_w_mul(m1,m2)<<endl;
//    cout<<m1*r1<<endl;
//    cout<<r1*m1<<endl;
//    cout<<scalar_mul(m1,2)<<endl;
//    cout<<scalar_div(m1,2)<<endl;
//    cout<<dot(r1,r2)<<endl;

//    Matrix<int> m2 = m1 + m1 + m1;
//    m2 = scalar_mul(m2, 3);
    // r1.push_back(1);
    // vector<int> c = cross(r1, r2);
    //   cout << m2 << endl;
    // cout << convolution(m1, m2);
}
