#include <iostream>
// #include <vector>
// #include <cmath>
// #include <numeric>
#include <ap_int.h>
#include <tapa.h>
#include <hls_vector.h>
#include <hls_math.h>
// #include <bits/stdc++.h>
#include <string>
#include <glog/logging.h>

using namespace std;

#define MEASURE_CYCLE_COUNT 1

#define N 256   // N is the sequence length
#define D 1024  // D is the dimension of the input
#define VEC_LEN 16
#define SCALE_FACTOR 32
// constexpr int norm = sqrt(D);

typedef ap_int<16> type_t;
typedef tapa::vec_t<type_t, VEC_LEN> vec_t;  // SIMD vector to use for the computation

void measure_cycle(tapa::istreams<bool, MEASURE_CYCLE_COUNT>& fifo_fin, tapa::mmap<int> cycle_count){
    measure_cycle_loop: for (int cycle = 0;;cycle++){
        bool flag_cont = false;
        for(int i = 0; i < MEASURE_CYCLE_COUNT; i++){
            flag_cont |= fifo_fin[i].empty();
        }
        if(!flag_cont){
            measure_cycle_loop_count: for (int i = 0; i < MEASURE_CYCLE_COUNT; i++){
                fifo_fin[i].read(nullptr);
            }
            cycle_count[0] = cycle;
            break;
        }
    }
}

// Kernel
void read_weight(
    tapa::async_mmap<vec_t>& vec,
    tapa::ostream<vec_t>& fifo_out,
    const std::string weight_name
){
    const int bound = D * D / VEC_LEN;
    for(int i_req = 0, i_resp = 0; i_resp < bound;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < bound) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        vec_t tmp_o;
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            // LOG(INFO) << "read " << weight_name << " vector: " << i_resp << " out of " << D * D / VEC_LEN - 1;
            i_resp++;
        }
    }
}

void read_input(
    tapa::async_mmap<vec_t>& vec,
    tapa::ostream<vec_t>& fifo_out,
    const std::string out_channel
){
    const int bound = N * D / VEC_LEN;
    for(int i_req = 0, i_resp = 0; i_resp < bound;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < bound) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        vec_t tmp_o;
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            // LOG(INFO) << "read " << out_channel << " vector: " << i_resp << " out of " << N * N / VEC_LEN - 1;
            i_resp++;
        }
    }
}

void write_output(
    tapa::async_mmap<vec_t>& output_mtx,
    tapa::istream<vec_t>& fifo_in,
    tapa::ostream<bool>& fifo_fin
){

    vec_t tmp;
    vec_t tmp_out;
    type_t col[N];

    for (int j = 0; j < D; j++){

        for(int i = 0; i < N / VEC_LEN;){
            if(!fifo_in.empty()){
                fifo_in.try_read(tmp);
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                    col[i * VEC_LEN + jj] = tmp[jj];
                }
                i++;
            }
        }

        for(int i_req = 0, i_resp = 0; i_resp < N / VEC_LEN;){
            if((i_req < N / VEC_LEN) & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
                for(int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
                    tmp_out[ii] = col[i_req * VEC_LEN + ii];
                }
                output_mtx.write_addr.try_write(j * N / VEC_LEN + i_req);
                output_mtx.write_data.try_write(tmp_out);
                i_req++;
            }
            bool success = false;
            auto resp = output_mtx.write_resp.read(success);
            if(success){
                i_resp += unsigned(resp)+1;
            }
        }

        LOG(INFO) << "Written Output to HBM column: " << j << " out of " << D-1;
    }

    fifo_fin.write(true);
} 


/**
 * @brief Projection of input to Q by doing X * Wq
 * 
 * This matrix multiplication using outer product approach to avoide caching the huge
 * Wq matrix entirely. Q is stored in row-wise manner where X is in column-wise manner.
 * 
 * This is the general projection function for Q, K, and V.
 * 
 * @param input_in_fifo streaming in X column by column
 * @param weight_in_fifo streaming in Wq row by row
 * @param output_out_fifo streaming out Q column by column
 */
void projection(
    tapa::istream<vec_t>& input_in_fifo, 
    tapa::istream<vec_t>& weight_in_fifo,
    tapa::ostream<vec_t>& output_out_fifo,
    const std::string projection_name
) {
    type_t result[N][D];  // break down result completely

    for(int i = 0; i < N; i++){
        for(int j = 0; j < D; j++){
            result[i][j] = 0;
        }
    }

    type_t input_col[N];
    type_t weight_row[D];
    vec_t tmp_input;
    vec_t tmp_weight;
    vec_t tmp_output;

    for (int k = 0; k < D; k++) {
#pragma HLS PIPELINE II=1
        // readin a column of input
        for (int i = 0; i < N / VEC_LEN;){
            if(!input_in_fifo.empty()){
                input_in_fifo.try_read(tmp_input);
                for(int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
                    input_col[i * VEC_LEN + ii] = tmp_input[ii];  // NOTE: Because VEC_LEN is power of 2, muliply is faster than add
                }
                i++;
            }
        }

        // readin a row of weight
        for (int j = 0; j < D / VEC_LEN;){
            if(!weight_in_fifo.empty()){
                weight_in_fifo.try_read(tmp_weight);
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                    weight_row[j * VEC_LEN + jj] = tmp_weight[jj];
                }
                j++;
            }
        }

        for (int i = 0; i < N; i++){
            for (int j = 0; j < D; j++){
                result[i][j] += input_col[i] * weight_row[j];
            }
        }
        // LOG(INFO) << "Computed " << projection_name << " partial sum: " << k << " out of " << D-1;
    }

    LOG(INFO) << "Computed " << projection_name;

    // stream out the result column by column
    for(int j = 0; j < D; j++){
        for(int i = 0; i < N / VEC_LEN;){
            for(int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
                tmp_output[ii] = result[i * VEC_LEN + ii][j];
            }
            bool success = output_out_fifo.try_write(tmp_output);
            if(success){
                i++;
            }
        }
        LOG(INFO) << "Written " << projection_name << " column: " << j << " out of " << D-1;
    }
}

/**
 * @brief Multiply Q and K^T result in S
 * 
 * @param q_in_fifo streaming in Q column by column
 * @param k_in_fifo streaming in K column by column
 * @param output_out_fifo streaming out S column by column
 */
void compute_S(
    tapa::istream<vec_t>& q_in_fifo, 
    tapa::istream<vec_t>& k_in_fifo,
    tapa::ostream<vec_t>& output_out_fifo,
    const std::string compute_S_name
) {
    type_t result[N][N];  // break down result completely

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            result[i][j] = 0;
        }
    }

    type_t q_col[N];
    type_t k_row[N];
    vec_t tmp_q;
    vec_t tmp_k;
    vec_t tmp_output;

    for (int k = 0; k < D; k++) {
#pragma HLS PIPELINE II=1
        // readin a column of input
        for (int i = 0; i < N / VEC_LEN;){
            if(!q_in_fifo.empty()){
                q_in_fifo.try_read(tmp_q);
                for(int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
                    q_col[i * VEC_LEN + ii] = tmp_q[ii];  // NOTE: Because VEC_LEN is power of 2, muliply is faster than add
                }
                i++;
            }
        }

        // readin a row of weight
        for (int j = 0; j < N / VEC_LEN;){
            if(!k_in_fifo.empty()){
                k_in_fifo.try_read(tmp_k);
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL  
                    k_row[j * VEC_LEN + jj] = tmp_k[jj];
                }
                j++;
            }
        }

        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                result[i][j] += q_col[i] * k_row[j];
            }
        }
        // LOG(INFO) << "Computed " << compute_S_name << " partial sum: " << k << " out of " << D-1;
    }

    LOG(INFO) << "Computed " << compute_S_name;

    // stream out the result row by row
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N / VEC_LEN;){
            for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                tmp_output[jj] = result[i][j * VEC_LEN + jj];
            }
            bool success = output_out_fifo.try_write(tmp_output);
            if(success){
                j++;
            }
        }
        LOG(INFO) << "Written " << compute_S_name << " row: " << i << " out of " << N-1;
    }
}



/**
 * @brief Multiply Softmax(QK^T/sqrt(d)) (S') and V result in Output
 * 
 * @param s_in_fifo streaming in S' column by column
 * @param v_in_fifo streaming in V column by column
 * @param output_out_fifo streaming out Output row by row
 */
void compute_output(
    tapa::istream<vec_t>& s_in_fifo, 
    tapa::istream<vec_t>& v_in_fifo,
    tapa::ostream<vec_t>& output_out_fifo
) {
    type_t result[N];  // break down result completely
    type_t S[N][N];  // cache S' because we need to reuse it's rows N times
    type_t v_col[N];

    vec_t tmp_v;
    vec_t tmp_s;
    vec_t tmp_out;  // temp output for steaming out the result. 

    // readin and cache S'
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N / VEC_LEN;){
            if(!s_in_fifo.empty()){
                s_in_fifo.try_read(tmp_s);
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                    S[i][j * VEC_LEN + jj] = tmp_s[jj];
                }
                j++;
            }
        }
    }

    LOG(INFO) << "Cached S\'";

    for (int j = 0; j < D; j++) {  // for column j in output
#pragma HLS PIPELINE II=1
        // initialize result[i]
        for(int i = 0; i < N; i++){
            result[i] = 0;
        }

        // readin a column of input
        for (int i = 0; i < N / VEC_LEN;){
            if(!v_in_fifo.empty()){
                v_in_fifo.try_read(tmp_v);
                for(int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
                    v_col[i * VEC_LEN + ii] = tmp_v[ii];  // NOTE: Because VEC_LEN is power of 2, muliply is faster than add
                }
                i++;
            }
        }

        for (int i = 0; i < N; i++){  // for row i in the output
            for (int k = 0; k < N; k++){
                result[i] += S[i][k] * v_col[k];
            }
            result[i] = result[i] / SCALE_FACTOR;  // scale down the result by SCALE_FACTOR
            // LOG(INFO) << "Computed " << "Output" << " (" << i << ", " << j << ") out of (" << N-1 << ", " << D-1 << ")";
        }

        // stream out the column
        for (int i = 0; i < N / VEC_LEN;){
            for(int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
                tmp_out[ii] = result[i * VEC_LEN + ii];
            }
            bool success = output_out_fifo.try_write(tmp_out);
            if(success){
                i++;
            }
        }

        LOG(INFO) << "Written " << "Output" << " column: " << j << " out of " << D-1;
    }
}

// Helper function to apply softmax row-wise on a matrix
void softmax(
    tapa::istream<vec_t>& input_in_fifo,
    tapa::ostream<vec_t>& output_out_fifo
) {
    type_t row[N];
    vec_t tmp_row;
    vec_t tmp_out;
    for(int i = 0; i < N; i++){
        type_t sum = 0;
        // readin a row of input
        for(int j = 0; j < N / VEC_LEN;){
            if(!input_in_fifo.empty()){
                input_in_fifo.try_read(tmp_row);
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                    row[j * VEC_LEN + jj] = tmp_row[jj];
                }
                j++;
            }
        }

        for (int j = 0; j < N; j++){
            row[j] = hls::exp(row[j]);
            sum += row[j];
        }
        for (int j = 0; j < N; j++){
            row[j] /= (sum / SCALE_FACTOR);
        }

        for(int j = 0; j < N / VEC_LEN;){
            for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                tmp_out[jj] = row[j * VEC_LEN + jj];
            }
            bool success = output_out_fifo.try_write(tmp_out);
            if(success){
                j++;
            }
        }

        LOG(INFO) << "Written " << "Softmax" << " row: " << i << " out of " << N-1;
    }
}

// Self-attention computation
void selfAttention(
    tapa::mmap<vec_t> top_input,
    tapa::mmap<vec_t> Wq, 
    tapa::mmap<vec_t> Wk, 
    tapa::mmap<vec_t> Wv, 
    tapa::mmap<vec_t> top_output,
    tapa::mmap<int> cycle_count
) {

    tapa::stream<vec_t> fifo_input_Q("fifo_input_Q");
    tapa::stream<vec_t> fifo_input_K("fifo_input_K");
    tapa::stream<vec_t> fifo_input_V("fifo_input_V");
    tapa::stream<vec_t> fifo_Wq("fifo_Wq");
    tapa::stream<vec_t> fifo_Wk("fifo_Wk");
    tapa::stream<vec_t> fifo_Wv("fifo_Wv");

    tapa::stream<vec_t> fifo_Q("fifo_Q");
    tapa::stream<vec_t> fifo_K("fifo_K");
    tapa::stream<vec_t> fifo_V("fifo_V");

    tapa::stream<vec_t> fifo_S("fifo_S");
    tapa::stream<vec_t> fifo_S_softmax("fifo_S_softmax");
    tapa::stream<vec_t> fifo_output("fifo_output");

    tapa::streams<bool, MEASURE_CYCLE_COUNT> fifo_fin("fifo_fin");

    // Step 1: Compute Query, Key, and Value matrices
    tapa::task()
        .invoke<tapa::join>(read_input, top_input, fifo_input_Q, "Projection_Q")  // read input and distribute to Q
        .invoke<tapa::join>(read_input, top_input, fifo_input_K, "Projection_K")  // read input and distribute to K
        .invoke<tapa::join>(read_input, top_input, fifo_input_V, "Projection_V")  // read input and distribute to V

        .invoke<tapa::join>(read_weight, Wq, fifo_Wq, "Wq")  // read Wq
        .invoke<tapa::join>(read_weight, Wk, fifo_Wk, "Wk")  // read Wk
        .invoke<tapa::join>(read_weight, Wv, fifo_Wv, "Wv")  // read Wv

        .invoke<tapa::join>(projection, fifo_input_Q, fifo_Wq, fifo_Q, "Q")  // Q = X * Wq
        .invoke<tapa::join>(projection, fifo_input_K, fifo_Wk, fifo_K, "K")  // K = X * Wk
        .invoke<tapa::join>(projection, fifo_input_V, fifo_Wv, fifo_V, "V")  // V = X * Wv
        .invoke<tapa::join>(compute_S, fifo_Q, fifo_K, fifo_S, "S")  // S = Q * K^T

        .invoke<tapa::join>(softmax, fifo_S, fifo_S_softmax)  // Softmax(S')

        .invoke<tapa::join>(compute_output, fifo_S_softmax, fifo_V, fifo_output)  // Output = Softmax(S') * V
        .invoke<tapa::join>(write_output, top_output, fifo_output, fifo_fin)  // write output to top_output
        
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);  // measure the cycle count
}
