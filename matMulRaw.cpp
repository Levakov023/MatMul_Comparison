#include <iostream>
#include <chrono>
#include <array>
#include <vector>

template <typename T, std::size_t Row, std::size_t Col>
using Array2d = std::array<std::array<T, Col>, Row>;

template <typename T, std::size_t Row, std::size_t Col>
void printArray(const Array2d<T, Row, Col>& arr) {
    for (const auto& arr_row : arr) {
        for (const auto& e: arr_row) //elems of the row
            std::cout << e << ' ';

        std::cout << '\n';
    }
}


std::vector<float> rawMatMul ( std::vector<float>& A, std::vector<float>& B,
    const int a_Rows,const int b_Columns, int K ) {
    std::vector<float> C (a_Rows * b_Columns, 0.0f);

    for (std::size_t i {0} ; i < a_Rows; i++) {
        for (std::size_t j {0} ; j < b_Columns; j++) {

            C[i * b_Columns + j] = 0;

            for (std::size_t z {0} ; z < K ; z++) {

                C[i * b_Columns + j] += A[i * K + z] * B [ z * b_Columns + j];
            }

        }
    }
    return C;
}

template <typename T>
void printVec( const std::vector<T>& vec ) {
    for (std::size_t i {0} ; i < std::size(vec); i++){ std::cout << vec[i] << ' ';}
}



int main() {
    int K = 1024;
    int a_rows = 718;
    int b_columns = 556;

    std::vector aVec(K*a_rows, 0.0f);
    std::vector bVec(K*b_columns, 0.0f);

    // Setting up and randomizing 2 vectors , same shape as Kuda test.

    for ( std::size_t c{0} ; c < std::size(aVec) ; c++) {
        aVec[c] = static_cast<float>(rand())/RAND_MAX ;
    }

    for ( std::size_t c{0} ; c < std::size(bVec) ; c++ ) {
        bVec[c] = static_cast<float>(rand()) / RAND_MAX ;
    }


    const auto startCpu = std::chrono::high_resolution_clock::now();

    std::vector c (rawMatMul(aVec, bVec, a_rows, b_columns, K));


    const auto endCpu = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<float> diff = endCpu - startCpu;
    std::cout << "Raw CPU time for same size matMul : " << diff.count() * 1000 << "ms\n";


    return 0;
}