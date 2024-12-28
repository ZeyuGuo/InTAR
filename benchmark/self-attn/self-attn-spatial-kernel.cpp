#include <iostream>
#include <ap_int.h>
#include <tapa.h>
#include <hls_vector.h>
#include <hls_math.h>
// #include <glog/logging.h>

using namespace std;

#define MEASURE_CYCLE_COUNT 1

#define N 256   // N is the sequence length
#define D 1024  // D is the dimension of the input
#define VEC_LEN 16
#define SCALE_FACTOR 32

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

/**
 * @brief Read weight matrix from HBM to FIFO row by row
 * 
 * @param vec HBM memory mapped to the weight matrix
 * @param fifo_out FIFO to stream out the weight matrix row by row
 */
void read_weight(
    tapa::async_mmap<vec_t>& vec,
    tapa::ostreams<vec_t, D / VEC_LEN>& fifo_out
){
    vec_t row[D / VEC_LEN];
    hls::vector<bool, D/VEC_LEN> written;

    for (int i = 0; i < D; i++){
        written = false;  // written is supposed to be a simd vector
        for(int i_req = 0, i_resp = 0; i_resp < D / VEC_LEN;){
            #pragma HLS pipeline II=1 style=stp
            if((i_req < D / VEC_LEN) & !vec.read_addr.full()){
                vec.read_addr.write(i * D / VEC_LEN + i_req);
                i_req++;
            }
            bool success = vec.read_data.try_read(row[i_resp]);
            if(success){
                // LOG(INFO) << "read " << weight_name << " vector: " << i * D / VEC_LEN + i_resp << " out of " << D * D / VEC_LEN - 1;
                i_resp++;
            }
        }

        // write the row to streams
        for (int j = 0; j < D / VEC_LEN;){
#pragma HLS UNROLL
            if(!written[j]){
                bool success = fifo_out[j].try_write(row[j]);
                if(success){
                    written[j] = true;
                    j++;
                }
            }
        }
    }

}

/**
 * @brief Read input matrix from HBM to FIFO column by column
 * 
 * @param vec HBM memory mapped to the input matrix
 * @param fifo_out FIFO to stream out the input matrix column by column
 */
void read_input(
    tapa::async_mmap<vec_t>& vec,
    tapa::ostreams<vec_t, N / VEC_LEN>& fifo_out
){
    vec_t col[N / VEC_LEN];
    hls::vector<bool, N/VEC_LEN> written;

    for (int i = 0; i < D; i++){
        written = false;  // written is supposed to be a simd vector
        for(int i_req = 0, i_resp = 0; i_resp < N / VEC_LEN;){
            #pragma HLS pipeline II=1 style=stp
            if((i_req < N / VEC_LEN) & !vec.read_addr.full()){
                vec.read_addr.write(i * N / VEC_LEN + i_req);
                i_req++;
            }
            bool success = vec.read_data.try_read(col[i_resp]);
            if(success){
                // LOG(INFO) << "read " << weight_name << " vector: " << i * N / VEC_LEN + i_resp << " out of " << N * N / VEC_LEN - 1;
                i_resp++;
            }
        }

        // write the row to streams
        for (int j = 0; j < N / VEC_LEN;){
#pragma HLS UNROLL
            if(!written[j]){
                bool success = fifo_out[j].try_write(col[j]);
                if(success){
                    written[j] = true;
                    j++;
                }
            }
        }
    }
}

/**
 * @brief Write output matrix to HBM from FIFO row by row
 * 
 * @param output_mtx HBM memory mapped to the output matrix
 * @param fifo_in FIFO to stream in the output matrix row by row
 * @param fifo_fin FIFO to signal the end of the computation
 */
void write_output(
    tapa::async_mmap<vec_t>& output_mtx,
    tapa::istreams<vec_t, D / VEC_LEN>& fifo_in,
    tapa::ostream<bool>& fifo_fin
){
    hls::vector<bool, D/VEC_LEN> read;
    vec_t row[D / VEC_LEN];

    for (int i = 0; i < N; i++){
        read = false;
        for (int j = 0; j < D / VEC_LEN;){
#pragma HLS UNROLL
            if(!read[j] && !fifo_in[j].empty()){
                vec_t tmp;
                fifo_in[j].read(tmp);
                row[j] = tmp;
                read[j] = true;
                j++;
            }
        }

        for(int j_req = 0, j_resp = 0; j_resp < D / VEC_LEN;){
            if((j_req < D / VEC_LEN) & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
                output_mtx.write_addr.try_write(i * D / VEC_LEN + j_req);
                output_mtx.write_data.try_write(row[j_resp]);
                j_req++;
            }
            bool success = false;
            auto resp = output_mtx.write_resp.read(success);
            if(success){
                j_resp += unsigned(resp)+1;
            }
        }
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
    tapa::istreams<vec_t, N / VEC_LEN>& input_in_fifo, 
    tapa::istreams<vec_t, D / VEC_LEN>& weight_in_fifo,
    tapa::ostreams<vec_t, N / VEC_LEN>& output_out_fifo
) {
    type_t result[N][D];  // break down result completely

    for(int i = 0; i < N; i++){
        for(int j = 0; j < D; j++){
            result[i][j] = 0;
        }
    }

    type_t input_col[N];
    type_t weight_row[D];
    hls::vector<bool, N/VEC_LEN> input_read;
    hls::vector<bool, D/VEC_LEN> weight_read;
    hls::vector<bool, N/VEC_LEN> output_write;

    for (int k = 0; k < D; k++) {
#pragma HLS PIPELINE II=1
        input_read = false;
        weight_read = false;

        // readin a column of input
        for (int i = 0; i < N / VEC_LEN;){
#pragma HLS UNROLL
            if(!input_in_fifo[i].empty() && !input_read[i]){
                vec_t tmp_input;
                input_in_fifo[i].read(tmp_input);
                for(int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
                    input_col[i * VEC_LEN + ii] = tmp_input[ii];  // NOTE: Because VEC_LEN is power of 2, muliply is faster than add
                }
                input_read[i] = true;
                i++;
            }
        }

        // readin a row of weight
        for (int j = 0; j < D / VEC_LEN;){
#pragma HLS UNROLL
            if(!weight_in_fifo[j].empty() && !weight_read[j]){
                vec_t tmp_weight;
                weight_in_fifo[j].read(tmp_weight);
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                    weight_row[j * VEC_LEN + jj] = tmp_weight[jj];
                }
                weight_read[j] = true;
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

    // stream out the result column by column
    for(int j = 0; j < D; j++){
        output_write = false;
        for(int i = 0; i < N / VEC_LEN;){
#pragma HLS UNROLL
            if(!output_write[i] && !output_out_fifo[i].full()){
                vec_t tmp_output;
                for(int ii = 0; ii < VEC_LEN; ii++){
                    tmp_output[ii] = result[i * VEC_LEN + ii][j];
                }
                bool success = output_out_fifo[i].try_write(tmp_output);
                if(success){
                    output_write[i] = true;
                    i++;
                }
            }
        }

        // LOG(INFO) << "Written " << projection_name << " column: " << j << " out of " << D-1;
    }
}

/**
 * @brief Multiply Q and K^T result in S
 * 
 * @param q_in_fifo streaming in Q column by column
 * @param k_in_fifo streaming in K column by column
 * @param output_out_fifo streaming out S row by row
 */
void compute_S(
    tapa::istreams<vec_t, N / VEC_LEN>& q_in_fifo, 
    tapa::istreams<vec_t, N / VEC_LEN>& k_in_fifo,
    tapa::ostreams<vec_t, N / VEC_LEN>& output_out_fifo
) {
    type_t result[N][N];  // break down result completely

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            result[i][j] = 0;
        }
    }

    type_t q_col[N];
    type_t k_row[N];
    hls::vector<bool, N/VEC_LEN> q_read;
    hls::vector<bool, N/VEC_LEN> k_read;
    hls::vector<bool, N/VEC_LEN> output_write;

    for (int k = 0; k < D; k++) {
#pragma HLS PIPELINE II=1
        q_read = false;
        k_read = false;
        // readin a column of input
        for (int i = 0; i < N / VEC_LEN;){
#pragma HLS UNROLL
            if(!q_in_fifo[i].empty() && !q_read[i]){
                vec_t tmp_q;
                q_in_fifo[i].read(tmp_q);
                for(int ii = 0; ii < VEC_LEN; ii++){
                    q_col[i * VEC_LEN + ii] = tmp_q[ii];  // NOTE: Because VEC_LEN is power of 2, muliply is faster than add
                }
                q_read[i] = true;
                i++;
            }
        }

        // readin a row of weight
        for (int j = 0; j < N / VEC_LEN;){
#pragma HLS UNROLL
            if(!k_in_fifo[j].empty() && !k_read[j]){
                vec_t tmp_k;
                k_in_fifo[j].read(tmp_k);
                for(int jj = 0; jj < VEC_LEN; jj++){
                    k_row[j * VEC_LEN + jj] = tmp_k[jj];
                }
                k_read[j] = true;
                j++;
            }
        }

        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                result[i][j] += q_col[i] * k_row[j];
            }
        }

        // LOG(INFO) << "Computed " << "S" << " partial sum: " << k << " out of " << D-1;
    }

    // stream out the result row by row
    for(int i = 0; i < N; i++){
        output_write = false;
        for(int j = 0; j < N / VEC_LEN;){
#pragma HLS UNROLL
            if(!output_write[j] && !output_out_fifo[j].full()){
                vec_t tmp_output;
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                    tmp_output[jj] = result[i][j * VEC_LEN + jj];
                }
                bool success = output_out_fifo[j].try_write(tmp_output);
                if(success){
                    output_write[j] = true;
                    j++;
                }
            }
        }

        // LOG(INFO) << "Written " << "S" << " row: " << i << " out of " << N-1;
    }
}



/**
 * @brief Multiply Softmax(QK^T/sqrt(d)) (S') and V result in Output
 * 
 * @param s_in_fifo streaming in S' row by row
 * @param v_in_fifo streaming in V column by column
 * @param output_out_fifo streaming out Output row by row
 */
void compute_output(
    tapa::istreams<vec_t, N / VEC_LEN>& s_in_fifo, 
    tapa::istreams<vec_t, N / VEC_LEN>& v_in_fifo,
    tapa::ostreams<vec_t, D / VEC_LEN>& output_out_fifo
) {
    type_t result[N][D];  // break down result completely
    type_t S[N][N];  // cache S' because we need to reuse it's rows N times

    for(int i = 0; i < N; i++){
        for(int j = 0; j < D; j++){
            result[i][j] = 0;
        }
    }

    type_t v_col[N];
    hls::vector<bool, N/VEC_LEN> v_read;
    hls::vector<bool, N/VEC_LEN> s_read;
    hls::vector<bool, D/VEC_LEN> output_write;

    // readin and cache S'
    for(int i = 0; i < N; i++){
        s_read = false;
        for(int j = 0; j < N / VEC_LEN;){
#pragma HLS UNROLL
            if(!s_in_fifo[j].empty() && !s_read[j]){
                vec_t tmp_s;
                s_in_fifo[j].read(tmp_s);
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                    S[i][j * VEC_LEN + jj] = tmp_s[jj];
                }
                s_read[j] = true;
                j++;
            }
        }
    }

    for (int j = 0; j < D; j++) {  // for column j in output
#pragma HLS PIPELINE II=1
        v_read = false;
        // readin a column of input
        for (int i = 0; i < N / VEC_LEN;){
#pragma HLS UNROLL
            if(!v_read[i] && !v_in_fifo[i].empty()){
                vec_t tmp_v;
                v_in_fifo[i].read(tmp_v);
                for(int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
                    v_col[i * VEC_LEN + ii] = tmp_v[ii];  // NOTE: Because VEC_LEN is power of 2, muliply is faster than add
                }
                v_read[i] = true;
                i++;
            }
        }

        for (int i = 0; i < N; i++){  // for row i in the output
            for (int k = 0; k < N; k++){
                result[i][j] += S[i][k] * v_col[k];
            }
            result[i][j] = result[i][j] / SCALE_FACTOR;  // scale down the result by SCALE_FACTOR
            // LOG(INFO) << "Computed " << "Output" << " (" << i << ", " << j << ") out of (" << N-1 << ", " << D-1 << ")";
        }
    }

    // stream out the result row by row
    for(int i = 0; i < N; i++){
        output_write = false;
        for(int j = 0; j < D / VEC_LEN;){
#pragma HLS UNROLL
            if(!output_write[j] && !output_out_fifo[j].full()){
                vec_t tmp_out;
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                    tmp_out[jj] = result[i][j * VEC_LEN + jj];
                }
                bool success = output_out_fifo[j].try_write(tmp_out);
                if(success){
                    output_write[j] = true;
                    j++;
                }
            }
        }

        // LOG(INFO) << "Written " << "Output" << " row: " << i << " out of " << N-1;
    }
}


/**
 * @brief Conduct a softmax operation on a row of input
 * 
 * @param input 
 * @param output 
 */
void softmax_row(
    const hls::vector<type_t, N>& input, 
    hls::vector<type_t, N>& output
) {
    type_t sum = 0.0;
    for (int i = 0; i < N; ++i) {
        output[i] = hls::exp(input[i]);
        sum += output[i];
    }
    for (int i = 0; i < N; i++){
        output[i] = output[i] / (sum / SCALE_FACTOR);  // scale up each output element by SCALE_FACTOR
    }
}

/**
 * @brief Straming in S row by row and compute the softmax of each row and output them row by row
 * 
 * @param s_in_fifo streaming in S row by row
 * @param output_out_fifo streaming out the softmax of each row
 */
void softmax(tapa::istreams<vec_t, N / VEC_LEN>& s_in_fifo, tapa::ostreams<vec_t, N / VEC_LEN>& output_out_fifo) {
    hls::vector<bool, N/VEC_LEN> read;
    hls::vector<bool, N/VEC_LEN> write;
    hls::vector<type_t, N> row;
    hls::vector<type_t, N> output;
    for (int i = 0; i < N; i++){
        read = false;
        // readin a row of S from streams
        for (int j = 0; j < N / VEC_LEN;){
#pragma HLS UNROLL
            if (!read[j] && !s_in_fifo[j].empty()){
                vec_t tmp;
                s_in_fifo[j].read(tmp);
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                    row[j * VEC_LEN + jj] = tmp[jj];
                }
                read[j] = true;
                j++;
            }
        }

        // LOG(INFO) << "Read " << "S" << " row: " << i << " out of " << N-1;

        // scale row
        for (int j = 0; j < N; j++){
            row[j] = row[j] / sqrt(D);
        }

        // apply softmax row-wise
        softmax_row(row, output);

        // LOG(INFO) << "Computed " << "Softmax of S" << " row: " << i << " out of " << N-1;

        // readin a row of output to streams
        for(int j = 0; j < N / VEC_LEN;){
            write = false;
#pragma HLS UNROLL
            if(!write[j] && !output_out_fifo[j].full()){
                vec_t tmp_output;
                for(int jj = 0; jj < VEC_LEN; jj++){
#pragma HLS UNROLL
                    tmp_output[jj] = output[j * VEC_LEN + jj];
                }
                bool success = output_out_fifo[j].try_write(tmp_output);
                if(success){
                    write[j] = true;
                    j++;
                }
            }
        }

        // LOG(INFO) << "Written " << "S\'" << " row: " << i << " out of " << N-1;
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

    tapa::streams<vec_t, N / VEC_LEN> fifo_input_Q("fifo_input_Q");
    tapa::streams<vec_t, N / VEC_LEN> fifo_input_K("fifo_input_K");
    tapa::streams<vec_t, N / VEC_LEN> fifo_input_V("fifo_input_V");
    tapa::streams<vec_t, D / VEC_LEN> fifo_Wq("fifo_Wq");
    tapa::streams<vec_t, D / VEC_LEN> fifo_Wk("fifo_Wk");
    tapa::streams<vec_t, D / VEC_LEN> fifo_Wv("fifo_Wv");

    tapa::streams<vec_t, N / VEC_LEN> fifo_Q("fifo_Q");
    tapa::streams<vec_t, N / VEC_LEN> fifo_K("fifo_K");
    tapa::streams<vec_t, N / VEC_LEN> fifo_V("fifo_V");

    tapa::streams<vec_t, N / VEC_LEN> fifo_S("fifo_S");
    tapa::streams<vec_t, N / VEC_LEN> fifo_S_softmax("fifo_S_softmax");

    tapa::streams<vec_t, D / VEC_LEN> fifo_output("fifo_output");

    tapa::streams<bool, MEASURE_CYCLE_COUNT> fifo_fin("fifo_fin");

    // Step 1: Compute Query, Key, and Value matrices
    tapa::task()
        .invoke<tapa::join>(read_input, top_input, fifo_input_Q)  // read input and distribute to Q
        .invoke<tapa::join>(read_input, top_input, fifo_input_K)  // read input and distribute to K
        .invoke<tapa::join>(read_input, top_input, fifo_input_V)  // read input and distribute to V

        .invoke<tapa::join>(read_weight, Wq, fifo_Wq)  // read Wq
        .invoke<tapa::join>(read_weight, Wk, fifo_Wk)  // read Wk
        .invoke<tapa::join>(read_weight, Wv, fifo_Wv)  // read Wv

        .invoke<tapa::join>(projection, fifo_input_Q, fifo_Wq, fifo_Q)  // Q = X * Wq
        .invoke<tapa::join>(projection, fifo_input_K, fifo_Wk, fifo_K)  // K = X * Wk
        .invoke<tapa::join>(projection, fifo_input_V, fifo_Wv, fifo_V)  // V = X * Wv
        .invoke<tapa::join>(compute_S, fifo_Q, fifo_K, fifo_S)  // S = Q * K^T

        .invoke<tapa::join>(softmax, fifo_S, fifo_S_softmax)  // S' = Softmax(S)

        .invoke<tapa::join>(compute_output, fifo_S_softmax, fifo_V, fifo_output)  // Output = Softmax(S') * V
        .invoke<tapa::join>(write_output, top_output, fifo_output, fifo_fin)  // write output to top_output
        
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);  // measure the cycle count
}

