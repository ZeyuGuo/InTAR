/*
FIXME: This host is from intrra host. 
*/

#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>
#include <cmath>
#include <ap_int.h>
#include "self-attn-sequential.h"

constexpr int seq_len = 256;
constexpr int D = 1024;
constexpr int D_head = 1024;
constexpr int D_head_div_2 = D_head / 2;
constexpr int weight_size_cc0 = D * D_head / 16;
constexpr int weight_size_cc1 = D * D_head / 8;
constexpr int input_size = seq_len * D / 16;
constexpr int output_size = seq_len * D_head / 32;

typedef ap_int<16> type_t;

int main(int argc, char *argv[]){
    // Example input and weight matrices

    std::cout << "TB Start" << std::endl;

    type_t input_[seq_len][D];
    type_t WQ_[D][D];
    type_t WK_[D][D];
    type_t WV_[D][D];
    type_t output_[seq_len][D];

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < D; j++) {
            input_[i][j] = (type_t) (rand() % 256);
            output_[i][j] = (type_t) 0;
        }
    }

    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            WQ_[i][j] = (type_t) (rand() % 256);
            WK_[i][j] = (type_t) (rand() % 256);
            WV_[i][j] = (type_t) (rand() % 256);
        }
    }

    std::cout << "TB Matrix Declared" << std::endl;

    selfAttention(input_, WQ_, WK_, WV_, output_);

    std::cout << "TB End" << std::endl;

    return 0;
}