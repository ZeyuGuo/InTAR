#include <iostream>
#include <cmath>
#include <numeric>
#include <string>
#include <tapa.h>
#include <ap_int.h>
#include <hls_math.h>

constexpr int batch_size = 32; // Number of input examples
constexpr int batch_size_div_2 = batch_size / 2;
constexpr int input_dim = 4096;  // Input dimension
constexpr int hidden_dim = 11008; // Hidden dimension
constexpr int weight_size = input_dim * hidden_dim / 8;
constexpr int input_size_cc0 = batch_size * input_dim;
constexpr int input_size_cc1 = batch_size * input_dim / 2;
constexpr int output_size = batch_size * input_dim; 

using int16_v16 = tapa::vec_t<ap_int<16>, 16>;

void read_W(
    const int w_size,
    tapa::async_mmap<int16_v16>& vec,
    tapa::ostream<int16_v16>& fifo_out
){
    for(int i_req = 0, i_resp = 0; i_resp < w_size;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < w_size) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        int16_v16 tmp_o; 
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

void read_X(
    const int i_size,
    tapa::async_mmap<int16_v16>& vec,
    tapa::ostream<int16_v16>& fifo_out
){
    for(int i_req = 0, i_resp = 0; i_resp < (i_size >> 4);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < (i_size >> 4)) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        int16_v16 tmp_o; 
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

void write_mtx(
    tapa::async_mmap<int16_v16>& output_mtx,
    tapa::istream<int16_v16>& fifo_in,
    tapa::ostream<bool>& fifo_fin
){

    for(int i_req = 0, i_resp = 0; i_resp < (output_size >> 4);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < (output_size >> 4)) & !fifo_in.empty() & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
            output_mtx.write_addr.try_write(i_req);
            int16_v16 tmp; fifo_in.try_read(tmp);
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

void CC0_up_gate(
    tapa::istream<int16_v16>& fifo_input,
    tapa::istream<int16_v16>& fifo_weight,
    tapa::ostream<int16_v16>& fifo_to_SFU,
    tapa::ostream<int16_v16>& fifo_to_CC1
) {
    ap_int<16> X[batch_size][input_dim];
    #pragma HLS array_partition variable=X complete dim=1

    for(int st = 0; st < 2; st++){

        const int fetch_bound = (batch_size_div_2 >> 4);
        const int send_bound = (st == 0) ? 1 : 2;

        for(int i = fetch_bound * st; i < fetch_bound * (st+1); i++){
            for(int j = 0; j < input_dim;){
                if(!fifo_input.empty()){
                    int16_v16 tmp; fifo_input.try_read(tmp);
                    for(int k = 0; k < 16; k++){
                        X[i*16+k][j] = tmp[k];
                    }
                    j++;
                }
            }
        }

        for(int i = 0; i < (hidden_dim >> 4); i++){
            ap_int<16> acc_vec[2][16][32];
            #pragma HLS array_partition variable=acc_vec complete dim=1
            #pragma HLS array_partition variable=acc_vec complete dim=2
            #pragma HLS array_partition variable=acc_vec complete dim=3
            
            for(int k = 0; k < 2; k++){
                #pragma HLS unroll
                for(int p = 0; p < 16; p++){
                    #pragma HLS unroll
                    for(int q = 0; q < 32; q++){
                        #pragma HLS unroll
                        acc_vec[k][p][q] = 0;
                    }
                }
            }

            for(int k = 0; k < input_dim; k++){
                #pragma HLS pipeline II=1
                int16_v16 tmp = fifo_weight.read();
                for(int p = 0; p < 16; p++){
                    for(int q = 0; q < 16; q++){
                        acc_vec[k % 2][p][q] += tmp[p] * X[q][k];
                    }
                }
                if(st == 1) {
                    for(int p = 0; p < 16; p++){
                        for(int q = 0; q < 16; q++){
                            acc_vec[k % 2][p][16+q] += tmp[p] * X[16+q][k];
                        }
                    }
                }
            }


            for(int p = 0; p < 16; p++){
                #pragma HLS unroll
                for(int q = 0; q < 16; q++){
                    #pragma HLS unroll
                    acc_vec[0][p][q] += acc_vec[1][p][q];
                }
            }

            if(st == 1) {
                for(int p = 0; p < 16; p++){
                    #pragma HLS unroll
                    for(int q = 0; q < 16; q++){
                        #pragma HLS unroll
                        acc_vec[0][p][16+q] += acc_vec[1][p][16+q];
                    }
                }
            }

            for(int m = 0; m < send_bound; m++){
                for(int k = 0; k < 16; k++){
                    #pragma HLS pipeline II=1
                    int16_v16 tmp;
                    for(int p = 0; p < 16; p++){
                        #pragma HLS unroll
                        tmp[p] = acc_vec[0][k][m*16+p];
                    }
                    if(st == 0) {
                        fifo_to_SFU.write(tmp);
                    } else {
                        fifo_to_CC1.write(tmp);
                    }
                }
            }

        }

    }

}

void CC1_down(
    tapa::istream<int16_v16>& fifo_input,
    tapa::istream<int16_v16>& fifo_weight,
    tapa::ostream<int16_v16>& fifo_to_SFU,
    tapa::istream<int16_v16>& fifo_from_CC0,
    tapa::ostream<int16_v16>& fifo_output
){
    ap_int<16> X[batch_size][input_dim];
    #pragma HLS array_partition variable=X complete dim=1

    for(int st = 0; st < 2; st++){

        const int ii_bound = (st == 0) ? 1 : (input_dim >> 4);
        const int jj_bound = (st == 0) ? 1 : 2;
        const int k_bound = (st == 0) ? input_dim : 16;

        if(st == 0){
            for(int i = 0; i < (batch_size_div_2 >> 4); i++){
                for(int j = 0; j < input_dim;){
                    if(!fifo_input.empty()){
                        int16_v16 tmp; fifo_input.try_read(tmp);
                        for(int k = 0; k < 16; k++){
                            X[i*16+k][j] = tmp[k];
                        }
                        j++;
                    }
                }
            }
        } else {
            for(int j = 0; j < input_dim; j++){
                for(int k = 0; k < 32; k++){
                    #pragma HLS unroll
                    X[k][j] = 0;
                }
            }
        }

        for(int i = 0; i < (hidden_dim >> 4); i++){
            ap_int<16> cache[batch_size][16];
            #pragma HLS array_partition variable=cache cyclic dim=1 factor=16
            if(st == 1){
                for(int i = 0; i < (batch_size >> 4); i++){
                    for(int j = 0; j < 16; j++){
                        int16_v16 tmp = fifo_from_CC0.read();
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            cache[i*16+k][j] = tmp[k];
                        }
                    }
                }
            }

            for(int ii = 0; ii < ii_bound; ii++){

                ap_int<16> cache_w[16][16];
                #pragma HLS array_partition variable=cache_w complete dim=2

                if(st == 1){
                    for(int j = 0; j < 16; j++){
                        int16_v16 tmp = fifo_weight.read();
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            cache_w[j][k] = tmp[k];
                        }
                    }
                }

                for(int jj = 0; jj < jj_bound; jj++){

                    ap_int<16> acc_vec[2][16][16];
                    #pragma HLS array_partition variable=acc_vec complete dim=1
                    #pragma HLS array_partition variable=acc_vec complete dim=2
                    #pragma HLS array_partition variable=acc_vec complete dim=3
                    
                    for(int k = 0; k < 2; k++){
                        #pragma HLS unroll
                        for(int p = 0; p < 16; p++){
                            #pragma HLS unroll
                            for(int q = 0; q < 16; q++){
                                #pragma HLS unroll
                                acc_vec[k][p][q] = 0;
                            }
                        }
                    }

                    for(int k = 0; k < k_bound; k++){
                        #pragma HLS pipeline II=1

                        ap_int<16> op1[16];
                        ap_int<16> op2[16];
                        #pragma HLS array_partition variable=op1 complete
                        #pragma HLS array_partition variable=op2 complete

                        int16_v16 tmp; 
                        if(st == 0) tmp = fifo_weight.read();

                        for(int p = 0; p < 16; p++){
                            #pragma HLS unroll
                            if(st == 0){
                                op1[p] = tmp[p];
                                op2[p] = X[p][k];
                            } else {
                                op1[p] = cache_w[k][p];
                                op2[p] = cache[jj*16+p][k];
                            }
                        }

                        for(int p = 0; p < 16; p++){
                            #pragma HLS unroll
                            for(int q = 0; q < 16; q++){
                                #pragma HLS unroll
                                acc_vec[k % 2][p][q] += op1[p] * op2[q];
                            }
                        }
                    }


                    for(int p = 0; p < 16; p++){
                        #pragma HLS unroll
                        for(int q = 0; q < 16; q++){
                            #pragma HLS unroll
                            acc_vec[0][p][q] += acc_vec[1][p][q];
                        }
                    }

                    if(st == 1) {
                        for(int p = 0; p < 16; p++){
                            #pragma HLS pipeline II=1
                            for(int q = 0; q < 16; q++){
                                #pragma HLS unroll
                                X[jj*16+q][ii*16+p] += acc_vec[0][p][q];
                            }
                        }
                    }

                    if(st == 0){
                        for(int k = 0; k < 16; k++){
                            #pragma HLS pipeline II=1
                            int16_v16 tmp;
                            for(int p = 0; p < 16; p++){
                                #pragma HLS unroll
                                tmp[p] = acc_vec[0][k][p];
                            }
                            fifo_to_SFU.write(tmp);
                        }
                    }
                }
            }
        }
    }

    for(int i = 0; i < (batch_size >> 4); i++){
        for(int j = 0; j < input_dim; j++){
            int16_v16 tmp;
            for(int k = 0; k < 16; k++){
                #pragma HLS unroll
                tmp[k] = X[i*16+k][j];
            }
            fifo_output.write(tmp);
        }
    }

}

void central_mem(
    tapa::istreams<int16_v16, 2>& fifo_from_CC,
    tapa::ostream<int16_v16>& fifo_to_CC0
){
    ap_int<16> hidden_cache[batch_size][hidden_dim];
    #pragma HLS array_partition variable=hidden_cache complete dim=1

    for(int i = 0; i < hidden_dim; i++){
        int16_v16 tmp0 = fifo_from_CC[0].read();
        int16_v16 tmp1 = fifo_from_CC[1].read();

        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            hidden_cache[j][i] = tmp0[j];
            hidden_cache[j+16][i] = tmp1[j];
        }
    }

    for(int i = 0; i < (hidden_dim >> 4); i++){
        for(int j = 0; j < (batch_size >> 4); j++){
            for(int k = 0; k < 16; k++){
                int16_v16 tmp;
                for(int l = 0; l < 16; l++){
                    #pragma HLS unroll
                    tmp[l] = hidden_cache[j*16+l][i*16+k];
                }
                fifo_to_CC0.write(tmp);
            }
        }
    }
}

void accumulate(
    tapa::istreams<int16_v16, 2>& fifo_in,
    tapa::ostream<int16_v16>& fifo_out
){
    for(;;){
        if(!fifo_in[0].empty() & !fifo_in[1].empty()){
            int16_v16 tmp0; fifo_in[0].try_read(tmp0);
            int16_v16 tmp1; fifo_in[1].try_read(tmp1);
            int16_v16 tmp_out;
            for(int i = 0; i < 16; i++){
                #pragma HLS unroll
                tmp_out[i] = tmp0[i] + tmp1[i];
            }
            fifo_out.write(tmp_out);
        }
    }
}

void SiLU(
    tapa::istream<int16_v16>& fifo_in,
    tapa::ostream<int16_v16>& fifo_out
){
    for(;;){
        if(!fifo_in.empty()){
            int16_v16 tmp; fifo_in.try_read(tmp);
            int16_v16 tmp_out;
            for(int i = 0; i < 16; i++){
                #pragma HLS unroll
                float denom = (float)1.0 / (float)(1.0 + hls::exp(-tmp[i]));
                float res = (float)(tmp[i]) * denom;
                tmp_out[i] = ap_int<16>((int)(res));
            }
            fifo_out.write(tmp_out);
        }
    }
}

void measure_cycle(tapa::istream<bool>& fifo_fin, tapa::mmap<int> cycle_count){
    for(int cycle = 0;;cycle++){
        if(!fifo_fin.empty()){
            fifo_fin.read(nullptr);
            cycle_count[0] = cycle;
            break;
        }
    }
}

// Forward pass through the FFN layer
void gatingNet(
    tapa::mmap<int16_v16> X_acc0,
    tapa::mmap<int16_v16> X_acc1,
    tapa::mmap<int16_v16> W_acc0,
    tapa::mmap<int16_v16> W_acc1,
    tapa::mmap<int16_v16> acc1_out,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<int16_v16> fifo_input_CC0("fifo_input_CC0");
    tapa::stream<int16_v16> fifo_input_CC1("fifo_input_CC1");
    tapa::stream<int16_v16> fifo_weight_CC0("fifo_weight_CC0");
    tapa::stream<int16_v16> fifo_weight_CC1("fifo_weight_CC1");

    tapa::streams<int16_v16, 2> fifo_to_SiLU("fifo_to_SiLU");
    tapa::streams<int16_v16, 2> fifo_to_central_mem("fifo_to_central_mem");
    tapa::streams<int16_v16, 2> fifo_to_acc("fifo_to_acc");
    tapa::stream<int16_v16, 128> fifo_to_CC1("fifo_to_CC1");

    tapa::stream<bool> fifo_fin("fifo_fin");
    tapa::stream<int16_v16> fifo_output("fifo_output");

    tapa::task()
        .invoke<tapa::join>(read_W, weight_size, W_acc0, fifo_weight_CC0)
        .invoke<tapa::join>(read_W, weight_size, W_acc1, fifo_weight_CC1)
        .invoke<tapa::join>(read_X, input_size_cc0, X_acc0, fifo_input_CC0)
        .invoke<tapa::join>(read_X, input_size_cc1, X_acc1, fifo_input_CC1)
        .invoke<tapa::join>(CC0_up_gate, fifo_input_CC0, fifo_weight_CC0, fifo_to_SiLU, fifo_to_acc)
        .invoke<tapa::join>(CC1_down, fifo_input_CC1, fifo_weight_CC1, fifo_to_SiLU, fifo_to_CC1, fifo_output)
        .invoke<tapa::detach, 2>(SiLU, fifo_to_SiLU, fifo_to_central_mem)
        .invoke<tapa::join>(central_mem, fifo_to_central_mem, fifo_to_acc)
        .invoke<tapa::detach>(accumulate, fifo_to_acc, fifo_to_CC1)
        .invoke<tapa::join>(write_mtx, acc1_out, fifo_output, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
    
}