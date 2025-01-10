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


constexpr int ID = 512;  // Input Dimension
constexpr int HD = 1376;  // Hidden Dimension
constexpr int B = 32;  // Batch Size
constexpr int VEC_LEN = 32;
constexpr int ID_div_VEC_LEN = ID / VEC_LEN;
constexpr int HD_div_VEC_LEN = HD / VEC_LEN;
constexpr int B_div_VEC_LEN = B / VEC_LEN;
typedef ap_int<64> type_t;
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
    vec_t tmp[VEC_LEN];

    for (int b = 0; b < B * ID_div_VEC_LEN / VEC_LEN; b++){
        for (int i = 0; i < VEC_LEN;){
            if(!fifo_in.empty()){
                bool success = fifo_in.try_read(tmp[i]);
                if(success){
                    i++;
                }
            }
        }

        for(int i_req = 0, i_resp = 0; i_resp < VEC_LEN;){

#pragma HLS pipeline II=1 style=stp
            if((i_req < VEC_LEN) & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
                int tile_row = b / ID_div_VEC_LEN;
                int tile_col = b % ID_div_VEC_LEN;
                int result_row = tile_row * VEC_LEN + i_req;  // or i_req?
                int result_col = tile_col;
                output_mtx.write_addr.try_write(result_row * ID_div_VEC_LEN + result_col);
                // printf("Request write output to addr %d\n", result_row * ID_div_VEC_LEN + result_col);
                output_mtx.write_data.try_write(tmp[i_req]);
                ++i_req;
            }
            bool write_success = false;
            auto resp = output_mtx.write_resp.read(write_success);
            if(write_success){
                i_resp += unsigned(resp)+1;
            } 
        }


    }
    


    // printf("Complete write output\n");

    fifo_fin.write(true);
    // printf("Write finish signal\n");
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
        vec_t tmp_out[B_div_VEC_LEN];

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

        up_matmul_row_tile_iter: for (int i = 0; i < B_div_VEC_LEN; i++){
            up_matmul_tile_row_iter:for (int ii = 0; ii < VEC_LEN; ii++){
                type_t c = 0;
                up_matmul_k_iter:for(int k = 0; k < ID_div_VEC_LEN; k++){
#pragma HLS pipeline II=1
                    vec_t a = input[i*VEC_LEN+ii][k];  // row i segment k of input 
                    vec_t b = weight_col[k];  // segment k of weight
                    up_matmul_l_iter: for (int l = 0; l < VEC_LEN; l++){
                        c += a[l] * b[l];
                    }
                }
                tmp_out[i][ii] = c;

                // printf("C value %lf\n", (double) c);

                // exit(0);
            }
        }

        for (int i = 0; i < B_div_VEC_LEN; i++){
            vec_t tmp_out_tmp = tmp_out[i];
            up_out_fifo.write(tmp_out_tmp);
            // printf("Wrote up projection result %d\n", j * B_div_VEC_LEN + i);
        }

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
    vec_t tmp_out[B_div_VEC_LEN]; vec_t tmp_out_tmp;
    vec_t up_tmp[B_div_VEC_LEN]; vec_t up_tmp_tmp;

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
// pragma HLS PIPELINE II=1

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
                bool success = up_in_fifo.try_read(up_tmp_tmp);
                if(success){
                    up_tmp[i] = up_tmp_tmp;
                    // printf("In gate read up projection result %d\n", j * B_div_VEC_LEN + i);
                    i++;
                }
            }
        }

        gate_matmul_row_tile_iter: for (int i = 0; i < B_div_VEC_LEN; i++){
#pragma HLS LOOP_TRIPCOUNT min=B_div_VEC_LEN max=B_div_VEC_LEN

            gate_matmul_tile_row_iter: for (int ii = 0; ii < VEC_LEN; ii++){
                type_t c = 0;
                gate_matmul_k_iter: for(int k = 0; k < ID_div_VEC_LEN; k++){
#pragma HLS pipeline II=1
                    vec_t a = input[i*VEC_LEN+ii][k];  // row i segment k of input 
                    vec_t b = weight_col[k];  // segment k of weight
                    gate_matmul_l_iter: for (int l = 0; l < VEC_LEN; l++){
                        c += a[l] * b[l];
                    }
                }
                tmp_out[i][ii] = c * up_tmp[i][ii];
            }
            // LOG(INFO) << "Wrote output " << j * HD_div_VEC_LEN + i << " in gate_projection";
        }

        gate_write_out: for (int i = 0; i < B_div_VEC_LEN; i++){
            vec_t tmp_out_tmp = tmp_out[i];
            output_out_fifo.write(tmp_out_tmp);
            // printf("Wrote gate projection result %d value %d and %f\n", j * B_div_VEC_LEN + i, tmp_out_tmp[0], tmp_out_tmp[VEC_LEN-1]);
        }
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
    vec_t combined_column[B_div_VEC_LEN]; vec_t combined_column_tmp;
    vec_t down_row[ID_div_VEC_LEN]; vec_t down_row_tmp;
    vec_t result[result_size]; vec_t tmp_result;

    // initialize the result matrix
    down_init_result: for (int i = 0; i < B; i++) {
        for (int j = 0; j < ID_div_VEC_LEN; j++) {
            for (int k = 0; k < VEC_LEN; k++) {
                result[i * ID_div_VEC_LEN + j][k] = 0;
            }
        }
    }

    // printf("Initialized down projection result\n");

    down_k_iter: for (int k = 0; k < HD; k++) {  // for column j in output
#pragma HLS LOOP_TRIPCOUNT min=HD max=HD

        // readin a row of down projection weight
        down_readin_row: for (int i = 0; i < ID_div_VEC_LEN;){  // FIXME: Those two loop readin the fifo in several cycles, not possible to make the outer loop II=1. 
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
                    combined_column[i] = combined_column_tmp;
                    i++;
                }
            }
        }

        // printf("Read combined result column %d\n", k);

        // LOG(INFO) << "Read combined result column " << k << " in down_projection";

        down_tile_row_iter: for (int i = 0; i < B_div_VEC_LEN; i++){  // tile iteration at vertical direction
            down_tile_col_iter: for (int j = 0; j < ID_div_VEC_LEN; j++){  // tile iteration at horizontal direction
                vec_t local_row = down_row[j];
                vec_t local_col = combined_column[i];
                type_t local_col_partitioned[VEC_LEN];
#pragma HLS ARRAY_PARTITION variable=local_col_partitioned complete dim=0
                for (int k = 0; k < VEC_LEN; k++){
#pragma HLS unroll
                    local_col_partitioned[k] = local_col[k];
                }

                int tile_offset = (i * ID_div_VEC_LEN + j) * VEC_LEN;

                down_tile_inner_row_iter: for (int ii = 0; ii < VEC_LEN; ii++){  // outer product form row wise
#pragma HLS unroll
                    vec_t result_row = result[tile_offset + ii];
                    result_row = result_row + local_col_partitioned[ii] * local_row;
//                     for (int jj = 0; jj < VEC_LEN; jj++){
// #pragma HLS unroll
//                         result_row[jj] = result_row[jj] + local_row[ii] * local_col[jj];
//                     }   
                    result[tile_offset + ii] = result_row;
                }
            }
        }

        // printf("Complete Down Projection Iteration %d\n", k);
    }

    // write the result to the output
    for (int i = 0; i < result_size; i++){
        output_out_fifo.write(result[i]);
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
                    // tmp_output[j] = tmp_input[j] / (1 + hls::exp(-tmp_input[j]));
                    tmp_output[j] = tmp_input[j];
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