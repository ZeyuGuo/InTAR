#include <iostream>
// #include <vector>
// #include <cmath>
// #include <numeric>
#include <ap_int.h>
#include <tapa.h>
#include <hls_vector.h>
#include <hls_math.h>
// #include <bits/stdc++.h>
// #include <string>
// #include <glog/logging.h>

using namespace std;

#define MEASURE_CYCLE_COUNT 1


constexpr int ID = 4096;  // Input Dimension
constexpr int HD = 11008;  // Hidden Dimension
constexpr int B = 32;  // Batch Size
constexpr int VEC_LEN = 32;
constexpr int ID_div_VEC_LEN = ID / VEC_LEN;
constexpr int HD_div_VEC_LEN = HD / VEC_LEN;
constexpr int B_div_VEC_LEN = B / VEC_LEN;
typedef ap_int<16> type_t;
typedef tapa::vec_t<type_t, VEC_LEN> vec_t;  // SIMD vector to use for the computation
constexpr int result_size = B * ID / VEC_LEN;
const int write_bound = B * ID / VEC_LEN;

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


/**
 * @brief Read the weight matrix column by column
 * 
 * @param vec 
 * @param fifo_out 
 */
void read_weight(
    tapa::async_mmap<vec_t>& vec,
    tapa::ostream<vec_t>& fifo_out
    // , const std::string weight_name
){
    const int bound = ID * HD / VEC_LEN;  // NOTE: W_down have different shape than W_up and W_gate but have same number of elements
    for(int i_req = 0, i_resp = 0; i_resp < bound;){
        #pragma HLS PIPELINE II=1 style=stp
        if((i_req < bound) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        vec_t tmp_o;
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

/**
 * @brief Read the input matrix row by row
 * 
 * @param vec 
 * @param fifo_out 
 */
void read_input(
    tapa::async_mmap<vec_t>& vec,
    tapa::ostream<vec_t>& fifo_out
    // , const std::string out_channel
){
    const int bound = B * ID / VEC_LEN;
    for(int i_req = 0, i_resp = 0; i_resp < bound;){
        #pragma HLS PIPELINE II=1 style=stp
        if((i_req < bound) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        vec_t tmp_o;
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

/**
 * @brief Write out the result in column major order
 * 
 * @param output_mtx 
 * @param fifo_in Result matrix are streamed in column by column
 * @param fifo_fin 
 */
void write_output(
    tapa::async_mmap<vec_t>& output_mtx,
    tapa::istream<vec_t>& fifo_in,
    tapa::ostream<bool>& fifo_fin
){
    for(int i_req = 0, i_resp = 0; i_resp < result_size;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < result_size) & !fifo_in.empty() & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
            output_mtx.write_addr.try_write(i_req);
            vec_t tmp; fifo_in.try_read(tmp);
            output_mtx.write_data.try_write(tmp);
            ++i_req;
        }
        bool success = false;
        auto resp = output_mtx.write_resp.read(success);
        if(success){
            i_resp += unsigned(resp)+1;
        }
    }

    fifo_fin.write(true);
} 


/**
 * @brief Projection of Input * W_up and Input * W_gate
 * 
 * This matrix multiplication using inner product approach. 
 * The weights are read from HBM row by row while the entire input is cached. 
 * The result is generated column by column. 
 * 
 * @param input_in_fifo streaming in X column by column
 * @param weight_in_fifo streaming in Wq row by row
 * @param output_out_fifo streaming out Q column by column
 */
void up_projection(
    tapa::istream<vec_t>& input_in_fifo, 
    tapa::istream<vec_t>& weight_in_fifo,
    tapa::ostream<vec_t>& up_out_fifo
) {
    vec_t input[B][ID_div_VEC_LEN];
    vec_t weight_col[ID_div_VEC_LEN];

#pragma HLS ARRAY_PARTITION variable=input type=complete dim=1

    // readin and cache the input matrix
    up_cache_input: for(int i = 0; i < B; i++){
        for(int j = 0; j < ID_div_VEC_LEN;){
            if(!input_in_fifo.empty()){
                vec_t tmp_vec; bool success = input_in_fifo.try_read(tmp_vec);
                if(success){
                    input[i][j] = tmp_vec;
                    j++;
                }
            }
        }
    }

    // LOG(INFO) << "Cached input in up_projection";

    // compute the result
    up_matmul_col_iter: for (int j = 0; j < HD; j++) {
#pragma HLS LOOP_TRIPCOUNT min=HD max=HD

        vec_t out;
        // readin a column of weight
        up_readin_col: for (int i = 0; i < ID_div_VEC_LEN;){
            if(!weight_in_fifo.empty()){
                vec_t tmp_vec; bool success = weight_in_fifo.try_read(tmp_vec);
                if(success){
                    weight_col[i] = tmp_vec;
                    // printf("Read up projection weight %d\n", j * ID_div_VEC_LEN + i);
                    i++;
                }
            }
        }

        type_t tmp_out[VEC_LEN]; 
#pragma HLS ARRAY_PARTITION variable=tmp_out type=complete dim=1

        up_matmul_tile_row_iter:for (int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL factor=8

            type_t c = 0;
            up_matmul_k_iter:for(int k = 0; k < ID_div_VEC_LEN; k++){
#pragma HLS pipeline II=1
                vec_t a = input[ii][k];  // row i segment k of input 
                vec_t b = weight_col[k];  // segment k of weight
                up_matmul_l_iter: for (int l = 0; l < VEC_LEN; l++){
                    c += a[l] * b[l];
                }
            }
            tmp_out[ii] = c;

            // printf("C value %lf\n", (double) c);

            // exit(0);
        }

        for (int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
            out[ii] = tmp_out[ii];
        }


        up_out_fifo.write(out);
        // printf("Wrote up projection column %d\n", j);
    }
}

/**
 * @brief Projection of Input * W_gate and multiply to the up projection on the fly. 
 * 
 * This matrix multiplication using inner product approach. 
 * The weights are read from HBM row by row while the entire input is cached. 
 * The result is generated column by column. 
 * 
 * @param input_in_fifo streaming in X column by column
 * @param weight_in_fifo streaming in Wq row by row
 * @param output_out_fifo streaming out Q column by column
 */
void gate_projection(
    tapa::istream<vec_t>& up_in_fifo,
    tapa::istream<vec_t>& input_in_fifo, 
    tapa::istream<vec_t>& weight_in_fifo,
    tapa::ostream<vec_t>& output_out_fifo
) {
    vec_t input[B][ID_div_VEC_LEN];
    vec_t weight_col[ID_div_VEC_LEN]; vec_t weight_col_tmp;
    vec_t tmp_input;
    vec_t out; type_t tmp_out[VEC_LEN];
    vec_t up; type_t up_tmp[VEC_LEN];

#pragma HLS ARRAY_PARTITION variable=input type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=tmp_out type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=up_tmp type=complete dim=1

    // readin and cache the input matrix
    gate_cache_input: for(int i = 0; i < B; i++){
        for(int j = 0; j < ID_div_VEC_LEN;){
            if(!input_in_fifo.empty()){
                vec_t tmp_vec; bool success = input_in_fifo.try_read(tmp_vec);
                if(success){
                    input[i][j] = tmp_vec;
                    j++;
                }
            }
        }
    }

    // LOG(INFO) << "Cached input in gate_projection";

    // compute the result
    gate_matmul_col_iter: for (int j = 0; j < HD; j++) {
#pragma HLS LOOP_TRIPCOUNT min=HD max=HD

        // readin a column of weight
        gate_readin_col: for (int i = 0; i < ID_div_VEC_LEN;){
            if(!weight_in_fifo.empty()){
                bool success = weight_in_fifo.try_read(weight_col_tmp);
                if(success){
                    weight_col[i] = weight_col_tmp;
                    i++;
                }
            }
        }

        gate_readin_up: for (int i = 0; i < B_div_VEC_LEN;){
            if (!up_in_fifo.empty()){
                bool success = up_in_fifo.try_read(up);
                if(success){
                    for (int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
                        up_tmp[ii] = up[ii];
                    }
                    i++;
                }
            }
        }

        gate_matmul_tile_row_iter: for (int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL factor=8

            type_t c = 0;
            gate_matmul_k_iter: for(int k = 0; k < ID_div_VEC_LEN; k++){
#pragma HLS pipeline II=1
                vec_t a = input[ii][k];  // row i segment k of input 
                vec_t b = weight_col[k];  // segment k of weight
                gate_matmul_l_iter: for (int l = 0; l < VEC_LEN; l++){
                    c += a[l] * b[l];
                }
            }
            tmp_out[ii] = c * up_tmp[ii];
        }
    
        for (int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
            out[ii] = tmp_out[ii];
        }

        output_out_fifo.write(out);

    }
}


/**
 * @brief Multiply combined with down projection weight with outer product approach
 * 
 * @param combined_in_fifo streaming in combined result column by column
 * @param down_in_fifo streaming in down projection weight column by column
 * @param output_out_fifo streaming out Output column by column
 */
void down_projection(
    tapa::istream<vec_t>& combined_in_fifo, 
    tapa::istream<vec_t>& down_in_fifo,
    tapa::ostream<vec_t>& output_out_fifo
) {
    type_t combined_column[VEC_LEN]; vec_t combined_column_tmp;
    vec_t down_row[ID_div_VEC_LEN]; vec_t down_row_tmp;
    vec_t result[B][ID_div_VEC_LEN]; vec_t tmp_result;

#pragma HLS ARRAY_PARTITION variable=result type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=combined_column type=complete dim=1

    // initialize the result matrix
    down_init_result: for (int i = 0; i < B; i++) {
        for (int j = 0; j < ID_div_VEC_LEN; j++) {
            result[i][j] = 0;
        }
    }
    // printf("Initialized down projection result\n");
    down_k_iter: for (int k = 0; k < HD; k++) { // for column j in output
#pragma HLS LOOP_TRIPCOUNT min=HD max=HD
        // readin a row of down projection weight
        down_readin_row: for (int i = 0; i < ID_div_VEC_LEN;){ // FIXME: Those two loop readin the fifo in several cycles, not possible to make the outer loop II=1. 
            if(!down_in_fifo.empty()){
                bool success = down_in_fifo.try_read(down_row_tmp);
                if(success){
                    down_row[i] = down_row_tmp;
                    i++;
                }
            }
        }
        // printf("Read down projection weight row %d\n", k);
        // LOG(INFO) << "Read down projection weight row " << k << " in down_projection";
        // readin a column of combined result
        down_readin_col: for (int i = 0; i < B_div_VEC_LEN;){
            if(!combined_in_fifo.empty()){
                bool success = combined_in_fifo.try_read(combined_column_tmp);
                if(success){
                    for (int ii = 0; ii < VEC_LEN; ii++){
#pragma HLS UNROLL
                        combined_column[ii] = combined_column_tmp[ii];
                    }
                    i++;
                }
            }
        }
        // printf("Read combined result column %d\n", k);
        // LOG(INFO) << "Read combined result column " << k << " in down_projection";
        down_tile_inner_row_iter: for (int i = 0; i < VEC_LEN; i++){
#pragma HLS unroll factor=8
            type_t col_val = combined_column[i];
            down_tile_col_iter: for (int j = 0; j < ID_div_VEC_LEN; j++){ // tile iteration at horizontal direction
#pragma HLS PIPELINE II=1
                vec_t local_row = down_row[j];
                vec_t result_row = result[i][j];
                for (int l = 0; l < VEC_LEN; l++){
#pragma HLS UNROLL
                    result_row[l] = result_row[l] + col_val * local_row[l];
                }
                result[i][j] = result_row;
            }
        }
        // printf("Complete Down Projection Iteration %d\n", k);
    }
    // write the result to the output
    for (int i = 0; i < B; i++){
        for (int j = 0; j < ID_div_VEC_LEN; j++){
            output_out_fifo.write(result[i][j]);
        }
        // printf("Wrote down projection result %d\n", i);
    }
}

/**
 * @brief Silu Function
 * 
 * Take in a vector of input, apply silu, then output. 
 * 
 * @param input_in_fifo 
 * @param output_out_fifo 
 */
void silu(
    tapa::istream<vec_t>& input_in_fifo,
    tapa::ostream<vec_t>& output_out_fifo
) {
    constexpr int silu_bound = B * HD / VEC_LEN;
    for (int i = 0; i < silu_bound;){
        if (!input_in_fifo.empty()){
            vec_t tmp_input; bool success = input_in_fifo.try_read(tmp_input);
            if(success){
                vec_t tmp_output;
                for (int j = 0; j < VEC_LEN; j++){
                    tmp_output[j] = tmp_input[j] / (1 + (type_t) hls::exp(-tmp_input[j]));
                    // tmp_output[j] = tmp_input[j];
                }
                output_out_fifo.write(tmp_output);
                i++;
            }
        }
    }
}

// Self-attention computation
void gating_net(
    tapa::mmap<vec_t> input_up,
    tapa::mmap<vec_t> input_gate,
    tapa::mmap<vec_t> W_up, 
    tapa::mmap<vec_t> W_gate, 
    tapa::mmap<vec_t> W_down, 
    tapa::mmap<vec_t> top_output,
    tapa::mmap<int> cycle_count
) {

    tapa::stream<vec_t> fifo_input_up("fifo_input_up");
    tapa::stream<vec_t> fifo_input_gate("fifo_input_gate");
    tapa::stream<vec_t> fifo_W_up("fifo_W_up");  // 512
    tapa::stream<vec_t> fifo_W_gate("fifo_W_gate");
    tapa::stream<vec_t> fifo_W_down("fifo_W_down");

    tapa::stream<vec_t> fifo_up("fifo_up");  // 512
    tapa::stream<vec_t> fifo_up_silu("fifo_up_silu");  // 512

    tapa::stream<vec_t> fifo_gate("fifo_gate");
    
    tapa::stream<vec_t> fifo_output("fifo_output");

    tapa::streams<bool, MEASURE_CYCLE_COUNT> fifo_fin("fifo_fin");

    // Step 1: Compute Query, Key, and Value matrices
    tapa::task()
        .invoke<tapa::join>(read_input, input_up, fifo_input_up)  // read input and distribute to up
        .invoke<tapa::join>(read_input, input_gate, fifo_input_gate)  // read input and distribute to gate

        .invoke<tapa::join>(read_weight, W_up, fifo_W_up)  // read W_up
        .invoke<tapa::join>(read_weight, W_gate, fifo_W_gate)  // read W_gate
        .invoke<tapa::join>(read_weight, W_down, fifo_W_down)  // read W_down

        .invoke<tapa::join>(up_projection, fifo_input_up, fifo_W_up, fifo_up)  // up = X * W_up
        .invoke<tapa::join>(silu, fifo_up, fifo_up_silu)
        .invoke<tapa::join>(gate_projection, fifo_up_silu, fifo_input_gate, fifo_W_gate, fifo_gate)  // gate = X * W_gate

        .invoke<tapa::join>(down_projection, fifo_gate, fifo_W_down, fifo_output)  // output = down * W_down
        .invoke<tapa::join>(write_output, top_output, fifo_output, fifo_fin)  // write output to top_output
        
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);  // measure the cycle count
}