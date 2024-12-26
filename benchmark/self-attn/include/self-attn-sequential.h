#include <iostream>
#include <ap_int.h>
#include <cmath>

using namespace std;

// Kernel
template <typename T, int M, int N>
void read_input(
    T input_mtx[M][N],
    T internal_bank[M][N]
){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            internal_bank[i][j] = input_mtx[i][j];
        }
    }
}

template <typename T, int M, int N>
void write_output(
    T output_mtx[M][N],
    T internal_bank[M][N]
){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            output_mtx[i][j] = internal_bank[i][j];
        }
    }
} 


// Helper function to apply softmax to a vector
template <typename T, int LEN>
void softmax(
    T input[LEN],
    T output[LEN]
) {
    float sum = 0.0;

    for (size_t i = 0; i < LEN; ++i) {
        output[i] = (T) exp(float(input[i]));
        sum += output[i];
    }
    for (size_t i = 0; i < LEN; ++i) {
        output[i] = output[i] / sum;
    }
}

// Helper function to apply softmax row-wise on a matrix
template <typename T, int M, int N> 
void softmax(
    T input_mtx[M][N],
    T output_mtx[M][N]
) {
    for (size_t i = 0; i < M; ++i) {
        softmax<T, N>(input_mtx[i], output_mtx[i]);
    }
}


template <typename T, int M, int K, int N>
void matMul(
    T A[M][K],
    T B[K][N],
    T C[M][N]
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS pipeline II=1
            C[i][j] = 0;
            for (int k = 0; k < K; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


// This is not to actually transpose the matrix, but to convert column major order to row major order. 
template <typename T, int M, int N>
void transpose(
    T input_mtx[M][N],
    T output_mtx[N][M]
) {
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            output_mtx[j][i] = input_mtx[i][j];
        }
    }
}

void selfAttention(
    ap_int<16> input_[256][1024],
    ap_int<16> WQ_[1024][1024],
    ap_int<16> WK_[1024][1024],
    ap_int<16> WV_[1024][1024],
    ap_int<16> output_[256][1024]
);