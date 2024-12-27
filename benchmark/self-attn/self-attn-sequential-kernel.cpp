#include <iostream>
#include <ap_int.h>
#include <tapa.h>
#include <glog/logging.h>

using namespace std;

#define N 256
#define D 1024
#define VEC_LEN 16

typedef ap_int<16> type_t;
using vec_t = tapa::vec_t<type_t, VEC_LEN>;

void measure_cycle(tapa::istreams<bool, 1>& fifo_fin, tapa::mmap<int> cycle_count){
    measure_cycle_loop: for (int cycle = 0;;cycle++){
        bool flag_cont = false;
        for(int i = 0; i < 1; i++){
            flag_cont |= fifo_fin[i].empty();
        }
        if(!flag_cont){
            measure_cycle_loop_count: for (int i = 0; i < 1; i++){
                fifo_fin[i].read(nullptr);
            }
            cycle_count[0] = cycle;
            break;
        }
    }
}

// Kernel
void read_input_N_D(
    type_t (&input_mtx)[N][D],
    type_t (&internal_bank)[N][D]
){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < D; j++){
            internal_bank[i][j] = input_mtx[i][j];
        }
    }
}

void read_input_D_D(
    type_t (&input_mtx)[D][D],
    type_t (&internal_bank)[D][D]
){
    for(int i = 0; i < D; i++){
        for(int j = 0; j < D; j++){
            internal_bank[i][j] = input_mtx[i][j];
        }
    }
}


void write_output(
    type_t (&output_mtx)[N][D],
    type_t (&internal_bank)[N][D]
){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < D; j++){
            output_mtx[i][j] = internal_bank[i][j];
        }
    }
} 


// Helper function to apply softmax to a vector
void softmax_row(
    type_t (&input)[N],
    type_t (&output)[N]
) {
    float sum = 0.0;

    for (size_t i = 0; i < N; ++i) {
        output[i] = (type_t) exp(float(input[i]));
        sum += output[i];
    }
    for (size_t i = 0; i < N; ++i) {
        output[i] = output[i] / sum;
    }
}

// Helper function to apply softmax row-wise on a matrix
void softmax(
    type_t (&input_mtx)[N][N],
    type_t (&output_mtx)[N][N]
) {
    for (size_t i = 0; i < N; ++i) {
        softmax_row(input_mtx[i], output_mtx[i]);
    }
}

void matMul_N_D_D(
    type_t (&A)[N][D],
    type_t (&B)[D][D],
    type_t (&C)[N][D]
) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
#pragma HLS pipeline II=1
            C[i][j] = 0;
            for (int k = 0; k < D; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matMul_N_D_N(
    type_t (&A)[N][D],
    type_t (&B)[D][N],
    type_t (&C)[N][N]
) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS pipeline II=1
            C[i][j] = 0;
            for (int k = 0; k < D; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matMul_N_N_D(
    type_t (&A)[N][N],
    type_t (&B)[N][D],
    type_t (&C)[N][D]
) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
#pragma HLS pipeline II=1
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// This is not to actually transpose the matrix, but to convert column major order to row major order. 
void transpose_N_D(
    type_t (&input_mtx)[N][D],
    type_t (&output_mtx)[D][N]
) {
    for(int i = 0; i < N; i++){
        for(int j = 0; j < D; j++){
            output_mtx[j][i] = input_mtx[i][j];
        }
    }
}

void scale(
    type_t (&input_mtx)[N][N],
    type_t (&output_mtx)[N][N]
) {
    type_t scale = sqrt(D);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            output_mtx[i][j] = input_mtx[i][j] / scale; // scaled attention
        }
    }
}

void vector_mac(const vec_t& a, const vec_t& b, type_t& c){
    for (int i = 0; i < VEC_LEN; i++){
        c += a[i] * b[i];
    }
}


void attention_top(
    tapa::async_mmap<vec_t>& in_glb,
    tapa::async_mmap<vec_t>& WQ_glb,  // flattened with column major order
    tapa::async_mmap<vec_t>& WK_glb,  // flattened with column major order
    tapa::async_mmap<vec_t>& WV_glb,  // flattened with column major order
    tapa::async_mmap<vec_t>& offchip_Q,
    tapa::async_mmap<vec_t>& offchip_K,
    tapa::async_mmap<vec_t>& offchip_V,
    tapa::async_mmap<vec_t>& out_glb,
    tapa::ostream<bool>& fifo_fin
){

    vec_t input[N * (D / VEC_LEN)];  // segment the input into block matrix with N by (D / VEC_LEN) blocks and flattened
    vec_t tmp_vec;

    type_t S[N][N];  // scores Q * K^T

    // Read and cache input
    const int input_size = N * D / VEC_LEN;
    read_input_loop: for(int i_req = 0, i_resp = 0; i_resp < input_size;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < input_size) & !in_glb.read_addr.full()){
            in_glb.read_addr.write(i_req);
            i_req++;
        }
        bool success = in_glb.read_data.try_read(tmp_vec);
        if(success){
            input[i_resp] = tmp_vec;
            i_resp++;
        }
    }

    LOG(INFO) << "Input read and cached";

    // Q = WQ * input
    for (int j = 0; j < D;){
        vec_t acc[D];
        for (int k_req = 0, k_resp = 0; k_resp < D / VEC_LEN;){
            if (k_req < D / VEC_LEN && !WQ_glb.read_addr.full()){
                WQ_glb.read_addr.write(j * D / VEC_LEN + k_req);
                k_req++;
                
            }
            bool success = WQ_glb.read_data.try_read(tmp_vec);
            if(success){                
                for (int i = 0; i < N; i++){  // input is already cached, can directly read
                    vector_mac(input[i * (D / VEC_LEN) + k_resp], tmp_vec, acc[i][j % VEC_LEN]);  // FIXME: Since VEC_LEN is 16, we can efficiently use the last 4 bits to index the vector for the mod operation. 
                }
                k_resp++;
            }
        }
        j++;  // increment first then use it to do the modulo, so we don't need to use (j_req + 1) % VEC_LEN
        if (j % VEC_LEN == 0){
            // write a column to offchip
            for (int i_req = 0, i_resp = 0; i_resp < N;){
                if (i_req < N && !offchip_Q.write_addr.full() && !offchip_Q.write_data.full()){
                    offchip_Q.write_addr.write(i_req * (D / VEC_LEN) + ((j-1) / VEC_LEN));
                    offchip_Q.write_data.try_write(acc[i_req]);
                    i_req++;
                }
                bool success = false;
                auto resp = offchip_Q.write_resp.read(success);
                if(success){
                    i_resp += unsigned(resp)+1;
                }
            }
        }
    }

    LOG(INFO) << "Q computed";

    // K = WK * input
    for (int j = 0; j < D;){
        vec_t acc[D];
        for (int k_req = 0, k_resp = 0; k_resp < D / VEC_LEN;){
            if (k_req < D / VEC_LEN && !WK_glb.read_addr.full()){
                WK_glb.read_addr.write(j * D / VEC_LEN + k_req);
                k_req++;
                
            }
            bool success = WK_glb.read_data.try_read(tmp_vec);
            if(success){                
                for (int i = 0; i < N; i++){  // input is already cached, can directly read
                    vector_mac(input[i * (D / VEC_LEN) + k_resp], tmp_vec, acc[i][j % VEC_LEN]);  // FIXME: Since VEC_LEN is 16, we can efficiently use the last 4 bits to index the vector for the mod operation. 
                }
                k_resp++;
            }
        }
        j++;  // increment first then use it to do the modulo, so we don't need to use (j_req + 1) % VEC_LEN
        if (j % VEC_LEN == 0){
            // write a column to offchip
            for (int i_req = 0, i_resp = 0; i_resp < N;){
                if (i_req < N && !offchip_K.write_addr.full() && !offchip_K.write_data.full()){
                    offchip_K.write_addr.write(i_req * (D / VEC_LEN) + ((j-1) / VEC_LEN));
                    offchip_K.write_data.try_write(acc[i_req]);
                    i_req++;
                }
                bool success = false;
                auto resp = offchip_K.write_resp.read(success);
                if(success){
                    i_resp += unsigned(resp)+1;
                }
            }
        }
    }

    LOG(INFO) << "K computed";

    // V = WV * input
    for (int j = 0; j < D;){
        vec_t acc[D];
        for (int k_req = 0, k_resp = 0; k_resp < D / VEC_LEN;){
            if (k_req < D / VEC_LEN && !WV_glb.read_addr.full()){
                WV_glb.read_addr.write(j * D / VEC_LEN + k_req);
                k_req++;
                
            }
            bool success = WV_glb.read_data.try_read(tmp_vec);
            if(success){                
                for (int i = 0; i < N; i++){  // input is already cached, can directly read
                    vector_mac(input[i * (D / VEC_LEN) + k_resp], tmp_vec, acc[i][j % VEC_LEN]);  // FIXME: Since VEC_LEN is 16, we can efficiently use the last 4 bits to index the vector for the mod operation. 
                }
                k_resp++;
            }
        }
        j++;  // increment first then use it to do the modulo, so we don't need to use (j_req + 1) % VEC_LEN
        if (j % VEC_LEN == 0){
            // write a column to offchip
            for (int i_req = 0, i_resp = 0; i_resp < N;){
                if (i_req < N && !offchip_V.write_addr.full() && !offchip_V.write_data.full()){
                    offchip_V.write_addr.write(i_req * (D / VEC_LEN) + ((j-1) / VEC_LEN));
                    offchip_V.write_data.try_write(acc[i_req]);
                    i_req++;
                }
                bool success = false;
                auto resp = offchip_V.write_resp.read(success);
                if(success){
                    i_resp += unsigned(resp)+1;
                }
            }
        }
    }

    LOG(INFO) << "V computed";

    // scores = Q * K^T
    vec_t tmp_q[D / VEC_LEN];
    vec_t tmp_k;
    for (int i = 0; i < N; i++){
        // first cache all the q row
        for (int kq_req = 0, kq_resp = 0; kq_resp < D / VEC_LEN;){
            if (kq_req < D / VEC_LEN && !offchip_Q.read_addr.full()){
                offchip_Q.read_addr.write(i * (D / VEC_LEN) + kq_req);
                kq_req++;
            }
            bool q_success = offchip_Q.read_data.try_read(tmp_q[kq_resp]);
            if (q_success){
                kq_resp++;
            }
        }
        
        for (int j = 0; j < N; j++){
            S[i][j] = 0;

            // accumulate according to the k column read
            for (int kk_req = 0, kk_resp = 0; kk_resp < D / VEC_LEN;){
                if (kk_req < D / VEC_LEN && !offchip_K.read_addr.full()){
                    offchip_K.read_addr.write(j * (D / VEC_LEN) + kk_req);
                    kk_req++;
                }
                bool k_success = offchip_K.read_data.try_read(tmp_k);
                if (k_success){
                    vector_mac(tmp_q[kk_resp], tmp_k, S[i][j]);
                    kk_resp++;
                }
            }
        }
    }

    LOG(INFO) << "S computed";

    // softmax(S)
    // softmax(S, S);
    // LOG(INFO) << "S softmaxed";

    // output = S * V
    
    for (int j = 0; j < D / VEC_LEN;){
        vec_t tmp_v[N];
        vec_t tmp_out[N];
        for (int i_req = 0, i_resp = 0; i_resp < N;){
            if (i_req < N && !offchip_V.read_addr.full()){
                offchip_V.read_addr.write(i_req * (D / VEC_LEN) + j);
                i_req++;
            }
            bool v_success = offchip_V.read_data.try_read(tmp_v[i_resp]);
            if (v_success){
                i_resp++;
            }
        }
        // compute a matrix mult between S and tmp_v with dimension N by N and N by D / VEC_LEN respectively
        for (int i = 0; i < N; i++){
            for (int j = 0; j < D / VEC_LEN; j++){
                tmp_out[i][j] = 0;
                for (int k = 0; k < N; k++){
                    tmp_out[i][j] += S[i][k] * tmp_v[k][j];
                }
            }
        }

        // write to output
        for (int i_req = 0, i_resp = 0; i_resp < N;){
            if (i_req < N && !out_glb.write_addr.full() && !out_glb.write_data.full()){
                out_glb.write_addr.write(i_req * (D / VEC_LEN) + j);
                out_glb.write_data.try_write(tmp_out[i_req]);
                i_req++;
            }
            bool success = false;
            auto resp = out_glb.write_resp.read(success);
            if(success){
                i_resp += unsigned(resp)+1;
            }
        }
        j++;
    }

    LOG(INFO) << "Output computed";
}

// Self-attention computation
void selfAttention(
    tapa::mmap<vec_t> in_glb,
    tapa::mmap<vec_t> WQ_glb,
    tapa::mmap<vec_t> WK_glb,
    tapa::mmap<vec_t> WV_glb,
    tapa::mmap<vec_t> offchip_Q,
    tapa::mmap<vec_t> offchip_K,
    tapa::mmap<vec_t> offchip_V,
    tapa::mmap<vec_t> out_glb,
    tapa::mmap<int> cycle_count
) {

    tapa::streams<bool, 1> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(
            attention_top,
            in_glb,
            WQ_glb,
            WK_glb,
            WV_glb,
            offchip_Q,
            offchip_K,
            offchip_V,
            out_glb,
            fifo_fin
        )
        .invoke<tapa::join>(
            measure_cycle,
            fifo_fin,
            cycle_count
        )
    ;
}
