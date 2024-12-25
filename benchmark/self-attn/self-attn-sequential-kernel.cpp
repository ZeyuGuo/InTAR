#include <iostream>
// #include <vector>
// #include <cmath>
// #include <numeric>
#include <ap_int.h>
#include <tapa.h>
// #include <bits/stdc++.h>

using namespace std;

// #define seq_len 256
// #define D 1024
// constexpr int norm = sqrt(D);

#define seq_len 32
#define D 128

typedef ap_int<16> type_t;
typedef tapa::vec_t<ap_int<16>, D> int16_vD;
typedef tapa::vec_t<ap_int<16>, seq_len> int16_vL;

// Kernel
void read_input_input(const int input_size,
    tapa::async_mmap<int16_vL>& vec,
    tapa::ostream<int16_vL>& fifo_out_wq,
    tapa::ostream<int16_vL>& fifo_out_wk,
    tapa::ostream<int16_vL>& fifo_out_wv
){
    const int bound = input_size >> 8;  // log2(seq_len = 256) = 8
    read_input_loop: for(int i_req = 0, i_resp = 0; i_resp < bound;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < bound) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        int16_vL tmp_o;
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out_wq.write(tmp_o);
            fifo_out_wk.write(tmp_o);
            fifo_out_wv.write(tmp_o);
            i_resp++;
        }
    }
}

void read_input_weights(
    const int input_size,
    tapa::async_mmap<int16_vD>& vec,
    tapa::ostream<int16_vD>& fifo_out
){
    const int bound = input_size >> 10;  // log2(D = 1024) = 10
    read_input_loop: for(int i_req = 0, i_resp = 0; i_resp < bound;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < bound) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        int16_vD tmp_o;
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

void write_mtx(
    tapa::async_mmap<int16_vD>& output_mtx,
    tapa::istream<int16_vD>& fifo_in
    // tapa::ostream<bool>& fifo_fin
){

    for(int i_req = 0, i_resp = 0; i_resp < seq_len;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < seq_len) & !fifo_in.empty() & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
            output_mtx.write_addr.try_write(i_req);
            int16_vD tmp; fifo_in.try_read(tmp);
            output_mtx.write_data.try_write(tmp);
            ++i_req;
        }
        bool success = false;
        auto resp = output_mtx.write_resp.read(success);
        if(success){
            i_resp += unsigned(resp)+1;
        }
    }

    // fifo_fin.write(true);
} 


// Helper function to perform matrix multiplication
void matMul(
    tapa::istream<int16_vL>& input_in_fifo, 
    tapa::istream<int16_vD>& weight_in_fifo,
    tapa::ostream<int16_vL>& output_out_fifo
) {
    size_t rows = seq_len;
    size_t cols = D;
    size_t common_dim = D;

    type_t result[seq_len][D];  // break down result completely
#pragma HLS ARRAY_PARTITION variable=result complete dim=0

    // result.assign(cols, hls::vector<type_t>(rows, 0));
    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < D; j++){
            result[i][j] = 0;
        }
    }

    for (int k = 0; k < 1;) {
// #pragma HLS LOOP_TRIPCOUNT min=c_size max=c_size
#pragma HLS PIPELINE II=1
        if (!input_in_fifo.empty() && !weight_in_fifo.empty()){
            int16_vL tmp_input; input_in_fifo.read(tmp_input);
            int16_vD tmp_weight; weight_in_fifo.read(tmp_weight);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result[i][j] += tmp_input[i] * tmp_weight[j];
                }
            }
            k++;
        }
    }

    // FIXME: reading individual elements from result which is breakdown to 
    // registers might cause large multiplexers to be generated and become non-synthesisable.
    for(int i = 0; i < D; i++){
        int16_vL tmp;
        for(int j = 0; j < seq_len; j++){
            tmp[j] = result[i][j];
        }
        output_out_fifo.write(tmp);
    }
}

void matMul_qk(  // FIXME: Special module for QK
    tapa::istream<int16_vL>& input_in_fifo, 
    tapa::istream<int16_vL>& weight_in_fifo,
    tapa::ostream<int16_vL>& output_out_fifo
) {
    size_t rows = seq_len;
    size_t cols = seq_len;
    size_t common_dim = D;

    type_t result[seq_len][seq_len];  // break down result completely
#pragma HLS ARRAY_PARTITION variable=result complete dim=0

    // result.assign(cols, hls::vector<type_t>(rows, 0));
    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < seq_len; j++){
            result[i][j] = 0;
        }
    }

    for (int k = 0; k < 1;) {  // FIXME: Why 1? This is referenced to row 90 in cnn-4L-spatial-kernel.cpp
// #pragma HLS LOOP_TRIPCOUNT min=c_size max=c_size
#pragma HLS PIPELINE II=1
        if (!input_in_fifo.empty() && !weight_in_fifo.empty()){
            int16_vL tmp_input; input_in_fifo.read(tmp_input);
            int16_vL tmp_weight; weight_in_fifo.read(tmp_weight);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result[i][j] += tmp_input[i] * tmp_weight[j];
                }
            }
            k++;
        }
    }

    // FIXME: reading individual elements from result which is breakdown to 
    // registers might cause large multiplexers to be generated and become non-synthesisable.
    for(int i = 0; i < seq_len; i++){
        int16_vL tmp;
        for(int j = 0; j < seq_len; j++){
            tmp[j] = result[i][j];
        }
        output_out_fifo.write(tmp);
    }
}


void matMul_v(  // FIXME: Special module when multiplying V and attention scores
    tapa::istream<int16_vL>& input_in_fifo, 
    tapa::istream<int16_vD>& weight_in_fifo,
    tapa::ostream<int16_vD>& output_out_fifo
) {
    size_t rows = seq_len;
    size_t cols = D;
    size_t common_dim = seq_len;

    type_t result[seq_len][D];  // break down result completely
#pragma HLS ARRAY_PARTITION variable=result complete dim=0

    // result.assign(cols, hls::vector<type_t>(rows, 0));
    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < D; j++){
            result[i][j] = 0;
        }
    }

    for (int k = 0; k < 1;) {
// #pragma HLS LOOP_TRIPCOUNT min=c_size max=c_size
#pragma HLS PIPELINE II=1
        if (!input_in_fifo.empty() && !weight_in_fifo.empty()){
            int16_vL tmp_input; input_in_fifo.read(tmp_input);
            int16_vD tmp_weight; weight_in_fifo.read(tmp_weight);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result[i][j] += tmp_input[i] * tmp_weight[j];
                }
            }
            k++;
        }
    }

    // FIXME: reading individual elements from result which is breakdown to 
    // registers might cause large multiplexers to be generated and become non-synthesisable.
    for(int i = 0; i < seq_len; i++){
        int16_vD tmp;
        for(int j = 0; j < D; j++){
            tmp[j] = result[i][j];
        }
        output_out_fifo.write(tmp);
    }
}


// // Helper function to apply softmax to a hls::vector
// void softmax(const hls::vector<float>& input, hls::vector<float>& output) {
//     output.resize(input.size());
//     float sum = 0.0;

//     for (size_t i = 0; i < input.size(); ++i) {
//         output[i] = exp(input[i]);
//         sum += output[i];
//     }
//     for (size_t i = 0; i < output.size(); ++i) {
//         output[i] /= sum;
//     }
// }

// // Helper function to apply softmax row-wise on a matrix
// void softmax(const Matrix& input, Matrix& output) {
//     output.resize(input.size());
//     for (size_t i = 0; i < input.size(); ++i) {
//         softmax(input[i], output[i]);
//     }
// }

// This is not to actually transpose the matrix, but to convert column major order to row major order. 
void transpose(
    tapa::istream<int16_vL>& input,
    tapa::ostream<int16_vD>& output
) {
    size_t rows = seq_len;
    size_t cols = D;
    int16_vD result[seq_len];
    for (size_t j = 0; j < cols; ++j) {
        int16_vL tmp; input.read(tmp);
        for (size_t i = 0; i < rows; ++i) {
            result[i][j] = tmp[i];
        }
    }

    for(int i = 0; i < seq_len; i++){
        output.write(result[i]);
    }
}

// Self-attention computation
void selfAttention(
    tapa::mmap<int16_vL> top_input,
    tapa::mmap<int16_vD> Wq, 
    tapa::mmap<int16_vD> Wk, 
    tapa::mmap<int16_vD> Wv, 
    tapa::mmap<int16_vD> top_output
) {

    tapa::stream<int16_vL> fifo_input_Wq("fifo_input_Wq");
    tapa::stream<int16_vL> fifo_input_Wk("fifo_input_Wk");
    tapa::stream<int16_vL> fifo_input_Wv("fifo_input_Wv");
    tapa::stream<int16_vD> fifo_Wq("fifo_Wq");
    tapa::stream<int16_vD> fifo_Wk("fifo_Wk");
    tapa::stream<int16_vD> fifo_Wv("fifo_Wv");

    tapa::stream<int16_vL> fifo_Q("fifo_Q");
    tapa::stream<int16_vL> fifo_K("fifo_K");
    tapa::stream<int16_vL> fifo_V("fifo_V");
    
    tapa::stream<int16_vL> fifo_QK("fifo_QK");
    tapa::stream<int16_vD> fifo_VT("fifo_VT");
    tapa::stream<int16_vD> fifo_output("fifo_output");

    // Step 1: Compute Query, Key, and Value matrices
    tapa::task()
        .invoke<tapa::join>(read_input_input, seq_len, top_input, fifo_input_Wq, fifo_input_Wk, fifo_input_Wv)
        .invoke<tapa::join>(read_input_weights, D, Wq, fifo_Wq)  // read Wq
        .invoke<tapa::join>(read_input_weights, D, Wk, fifo_Wk)  // read Wk
        .invoke<tapa::join>(read_input_weights, D, Wv, fifo_Wv)  // read Wv
        .invoke<tapa::join>(matMul, fifo_input_Wq, fifo_Wq, fifo_Q)
        .invoke<tapa::join>(matMul, fifo_input_Wk, fifo_Wk, fifo_K)
        .invoke<tapa::join>(matMul, fifo_input_Wv, fifo_Wv, fifo_V)
        .invoke<tapa::join>(matMul_qk, fifo_Q, fifo_K, fifo_QK)
        .invoke<tapa::join>(transpose, fifo_V, fifo_VT)
        .invoke<tapa::join>(matMul_v, fifo_QK, fifo_VT, fifo_output)
        .invoke<tapa::join>(write_mtx, top_output, fifo_output);
}

