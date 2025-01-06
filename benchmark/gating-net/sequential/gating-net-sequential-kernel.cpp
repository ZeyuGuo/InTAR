#include <iostream>
#include <ap_int.h>
#include <tapa.h>
// #include <glog/logging.h>

using namespace std;

#define B 8
#define ID 4096
#define HD 11008
#define VEC_LEN 8
#define SCALE_FACTOR 32

constexpr int ID_div_VEC_LEN = ID / VEC_LEN;
constexpr int HD_div_VEC_LEN = HD / VEC_LEN;
constexpr int B_div_VEC_LEN = B / VEC_LEN;

typedef ap_int<16> type_t;
using vec_t = tapa::vec_t<type_t, VEC_LEN>;
constexpr int input_size = B * ID / VEC_LEN;

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

void vector_mac(const vec_t& a, const vec_t& b, type_t& c){
#pragma HLS inline
    for (int i = 0; i < VEC_LEN; i++){
        c += a[i] * b[i];
    }
}

void gating_net_top(
    tapa::async_mmap<vec_t>& input,
    tapa::async_mmap<vec_t>& W_up,
    tapa::async_mmap<vec_t>& W_gate,
    tapa::async_mmap<vec_t>& W_down,
    tapa::async_mmap<vec_t>& output,
    tapa::ostream<bool>& fifo_fin
) {
    vec_t input_cache[input_size];
    vec_t up_result[B][HD_div_VEC_LEN];
    vec_t gate_result[B][HD_div_VEC_LEN];
    vec_t col_W[ID_div_VEC_LEN];
    vec_t tmp_out[B];
    vec_t tmp_vec;

#pragma HLS bind_storage variable=input_cache type=ram_2p impl=uram
#pragma HLS bind_storage variable=up_result type=ram_2p impl=uram
#pragma HLS bind_storage variable=gate_result type=ram_2p impl=uram
#pragma HLS bind_storage variable=tmp_out type=ram_2p impl=bram

    // Read and cache input
    read_input_up_loop: for(int i_req = 0, i_resp = 0; i_resp < input_size;) {
        #pragma HLS pipeline II=1 style=stp
        if((i_req < input_size) & !input_up.read_addr.full()) {
            input_up.read_addr.write(i_req);
            i_req++;
        }
        bool success = input_up.read_data.try_read(tmp_vec);
        if(success) {
            input_up_cache[i_resp] = tmp_vec;
            i_resp++;
        }
    }

    // Compute up path
    for (int j = 0; j < HD; j++) {
        // readin a column of W_up
        for (int k_req = 0, k_resp = 0; k_resp < ID_div_VEC_LEN;) {
            if (k_req < ID_div_VEC_LEN && !W_up.read_addr.full()) {
                W_up.read_addr.write(j * ID_div_VEC_LEN + k_req);
                k_req++;
            }
            bool success = W_up.read_data.try_read(col_W[k_resp]);
            if(success) {
                k_resp++;
            }
        }

        // compute a column
        for (int i = 0; i < B; i++) {
            for (int k = 0; k < ID_div_VEC_LEN; k++) {
#pragma HLS PIPELIN II=1 style=stp
                type_t acc = 0;
                for (int kk = 0; kk < VEC_LEN; kk++) {
                    acc += input_cache[i * (ID / VEC_LEN) + k][kk] * col_W[k][kk];
                }
                up_result[i][j / VEC_LEN][j % VEC_LEN] = acc;
            }
        }
    }

    // Compute gate path
    for (int j = 0; j < HD; j++) {
        // readin a column of W_up
        for (int k_req = 0, k_resp = 0; k_resp < ID_div_VEC_LEN;) {
            if (k_req < ID_div_VEC_LEN && !W_gate.read_addr.full()) {
                W_gate.read_addr.write(j * ID_div_VEC_LEN + k_req);
                k_req++;
            }
            bool success = W_gate.read_data.try_read(col_W[k_resp]);
            if(success) {
                k_resp++;
            }
        }

        // compute a column
        for (int i = 0; i < B; i++) {
            for (int k = 0; k < ID_div_VEC_LEN; k++) {
#pragma HLS PIPELIN II=1 style=stp
                type_t acc = 0;
                for (int kk = 0; kk < VEC_LEN; kk++) {
                    acc += input_cache[i * (ID / VEC_LEN) + k][kk] * col_W[k][kk];
                }
                gate_result[i][j / VEC_LEN][j % VEC_LEN] = acc;
            }
        }
    }

    // Compute silu for up_result in-place
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < HD_div_VEC_LEN; j++) {
            vec_t acc = up_result[i][j];
#pragma HLS PIPELIN II=1 style=stp
            for (int jj = 0; jj < VEC_LEN; jj++) {
                acc[jj] = acc[jj] / (1 + exp(-acc[jj]));
            }
            up_result[i][j] = acc;
        }
    }

    // compute summation of up_result and gate_result inplace
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < HD_div_VEC_LEN; j++) {
#pragma HLS PIPELIN II=1 style=stp
            vec_t acc = up_result[i][j];
            vec_t gate_seg = gate_result[i][j];
            for (int jj = 0; jj < VEC_LEN; jj++) {
                acc[jj] += gate_seg[jj];
            }
            up_result[i][j] = acc;
        }
    }

    // Down projection and output column by column
    for (int j = 0; j < ID; j++) {
        vec_t tmp_col[B_div_VEC_LEN];
        for (int i_req = 0, i_resp = 0; i_resp < B_div_VEC_LEN;) {
            if (i_req < B_div_VEC_LEN && !W_down.read_addr.full()) {
                W_down.read_addr.write(j * B_div_VEC_LEN + i_req);
                i_req++;
            }
            bool success = W_down.read_data.try_read(tmp_col[i_resp]);
            if(success) {
                i_resp++;
            }
        }

        // compute
        for (int i = 0; i < B; i++) {
            for (int k = 0; k < B_div_VEC_LEN; k++) {
#pragma HLS PIPELIN II=1 style=stp
                type_t acc = 0;
                vec_t up_seg = up_result[i][k];
                vec_t down_seg = tmp_col[k];
                for (int kk = 0; kk < VEC_LEN; kk++) {
                    acc += up_seg[kk] * down_seg[kk];
                }
                tmp_out[i][j % VEC_LEN] = acc;
            }
        }

        if ((j + 1) % VEC_LEN == 0) {
            // Write output tile by tile
            for (int i_req = 0, i_resp = 0; i_resp < B;) {
                if (i_req < B && !output.write_addr.full() && !output.write_data.full()) {
                    output.write_addr.write(i_req * (ID / VEC_LEN) + j / VEC_LEN);
                    output.write_data.try_write(tmp_out[i_req]);
                    i_req++;
                }
                bool success = false;
                auto resp = output.write_resp.read(success);
                if(success) {
                    i_resp += unsigned(resp)+1;
                }
            }
        }


    }

    fifo_fin.write(true);
}

void gating_net(
    tapa::mmap<vec_t> input,
    tapa::mmap<vec_t> W_up,
    tapa::mmap<vec_t> W_gate,
    tapa::mmap<vec_t> W_down,
    tapa::mmap<vec_t> output,
    tapa::mmap<int> cycle_count
) {
    tapa::streams<bool, 1> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(
            gating_net_top,
            input_up,
            input_gate,
            W_up,
            W_gate,
            W_down,
            output,
            fifo_fin
        )
        .invoke<tapa::join>(
            measure_cycle,
            fifo_fin,
            cycle_count
        )
    ;
}
