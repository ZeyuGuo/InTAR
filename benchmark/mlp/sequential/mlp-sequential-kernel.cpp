#include <iostream>
#include <cmath>
#include <numeric>
#include <string>
#include <tapa.h>
#include <ap_int.h>
#include <hls_math.h>

constexpr int input_size = 256;     // Number of input features
constexpr int hidden_size1 = 1024;   // Number of neurons in the first hidden layer
constexpr int hidden_size2 = 2048;   // Number of neurons in the second hidden layer
constexpr int output_size = 256;    // Number of output classes

constexpr int weight_size1 = input_size * hidden_size1 / 16;
constexpr int weight_size2 = hidden_size1 * hidden_size2 / 16;
constexpr int weight_size3 = hidden_size2 * output_size / 16;

using int16_v16 = tapa::vec_t<ap_int<16>, 16>;

void measure_cycle(tapa::istream<bool>& fifo_fin, tapa::mmap<int> cycle_count){
    for(int cycle = 0;;cycle++){
        if(!fifo_fin.empty()){
            fifo_fin.read(nullptr);
            cycle_count[0] = cycle;
            break;
        }
    }
}

void top(
    tapa::async_mmap<int16_v16>& X,
    tapa::async_mmap<int16_v16>& W1,
    tapa::async_mmap<int16_v16>& W2,
    tapa::async_mmap<int16_v16>& W3,
    tapa::async_mmap<int16_v16>& data_out,
    tapa::ostream<bool>& fifo_fin
) {
    ap_int<16> tmp1[hidden_size1];
    ap_int<16> tmp2[hidden_size2];
    #pragma HLS array_partition variable=tmp1 cyclic factor=16
    #pragma HLS array_partition variable=tmp2 cyclic factor=16

    int16_v16 pkt;

    // Read X
    for (int i_req = 0, i_resp = 0; i_resp < (input_size >> 4);) {
        #pragma HLS pipeline II=1 style=stp
        if((i_req < (input_size >> 4)) & !X.read_addr.full()){
            X.read_addr.write(i_req);
            i_req++;
        }
        int16_v16 tmp_o;
        bool success = X.read_data.try_read(tmp_o);
        if (success) {
            for (int i = 0; i < 16; i++) {
                #pragma HLS unroll
                tmp2[i_resp*16 + i] = tmp_o[i];
            }
            i_resp++;
        }
    }
    // Compute layer 1
    for (int i = 0; i < (hidden_size1 >> 4); i++) {
        int16_v16 sum;
        layer1_inner: for (int j = 0; j < input_size; j++) {
            #pragma HLS latency max=2
            for (;W1.read_addr.full();) { }
            W1.read_addr.write(i*input_size + j);
            for (;W1.read_data.empty();) { }
            int16_v16 tmp = W1.read_data.read(nullptr);

            for (int k = 0; k < 2; k++) {
                #pragma HLS pipeline II=1
                for (int kk = 0; kk < 8; kk++) {
                    #pragma HLS unroll
                    sum[k*8 + kk] = tmp2[j] * tmp[k*8 + kk];
                }
            }
        }
        for (int k = 0; k < 16; k++) {
            tmp1[i*16 + k] = sum[k];
        }
    }
    // Compute layer 2
    for (int i = 0; i < (hidden_size2 >> 4); i++) {
        int16_v16 sum;
        layer2_inner: for (int j = 0; j < hidden_size1; j++) {
            #pragma HLS latency max=2
            for (;W2.read_addr.full();) { }
            W2.read_addr.write(i*hidden_size1+ j);
            for (;W2.read_data.empty();) { }
            int16_v16 tmp = W2.read_data.read(nullptr);

            for (int k = 0; k < 2; k++) {
                #pragma HLS pipeline II=1
                for (int kk = 0; kk < 8; kk++) {
                    #pragma HLS unroll
                    sum[k*8 + kk] = tmp1[j] * tmp[k*8 + kk];
                }
            }
        }
        for (int k = 0; k < 16; k++) {
            tmp2[i*16 + k] = sum[k];
        }
    }
    // Compute layer 3
    for (int i = 0; i < (output_size >> 4); i++) {
        int16_v16 sum;
        layer3_inner: for (int j = 0; j < hidden_size2; j++) {
            #pragma HLS latency max=2
            for (;W3.read_addr.full();) { }
            W3.read_addr.write(i*hidden_size2 + j);
            for (;W3.read_data.empty();) { }
            int16_v16 tmp = W3.read_data.read(nullptr);

            for (int k = 0; k < 2; k++) {
                #pragma HLS pipeline II=1
                for (int kk = 0; kk < 8; kk++) {
                    #pragma HLS unroll
                    sum[k*8 + kk] = tmp2[j] * tmp[k*8 + kk];
                }
            }
        }
        for (;data_out.write_addr.full() || data_out.write_data.full();) { }
        data_out.write_addr.write(i);
        data_out.write_data.write(sum);
        for (;data_out.write_resp.empty();) { }
        data_out.write_resp.read(nullptr);
    }

    fifo_fin.write(true);
}

void MLP(
    tapa::mmap<int16_v16> X,
    tapa::mmap<int16_v16> W1,
    tapa::mmap<int16_v16> W2,
    tapa::mmap<int16_v16> W3,
    tapa::mmap<int16_v16> data_out,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(top, X, W1, W2, W3, data_out, fifo_fin)

        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count)
    ;
}
