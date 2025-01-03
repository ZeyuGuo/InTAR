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
    tapa::async_mmap<int16_v16>& data_in,
    tapa::async_mmap<int16_v16>& W1,
    tapa::async_mmap<int16_v16>& W2,
    tapa::async_mmap<int16_v16>& W3,
    tapa::async_mmap<int16_v16>& offchip,
    tapa::async_mmap<int16_v16>& data_out,
    tapa::ostream<bool>& fifo_fin
) {
    int16_v16 X[(input_size >> 4)];
    int16_v16 scratchpad[(hidden_size1 >> 4)];
    #pragma HLS array_partition variable=X cyclic factor=16
    #pragma HLS array_partition variable=scratchpad cyclic factor=16

    int16_v16 pkt;

    // Read X
    read_X: for (int i_req = 0, i_resp = 0; i_resp < (input_size >> 4);) {
        #pragma HLS pipeline II=1 style=stp
        if((i_req < (input_size >> 4)) & !data_in.read_addr.full()){
            data_in.read_addr.write(i_req);
            i_req++;
        }
        int16_v16 tmp_o;
        bool success = data_in.read_data.try_read(tmp_o);
        if (success) {
            X[i_resp] = tmp_o;
            i_resp++;
        }
    }

    // Compute 3 layers
    compute_layers: for (int layer = 0; layer < 3; layer++) {
        int bound_i = (layer == 0) ? (hidden_size1 >> 4) : (hidden_size2 >> 4);
        bound_i = (layer == 2) ? (output_size >> 4) : bound_i;

        int bound_j = (layer == 0) ? (input_size >> 4) : (hidden_size1 >> 4);
        bound_j = (layer == 2) ? (hidden_size2 >> 4) : bound_j;

        layer_outer: for (int i = 0; i < bound_i; i++) {
            int16_v16 sum;
            layer_inner: for (int j = 0; j < bound_j; j++) {
                int16_v16 op1;


                // Read from offchip for layer 3
                if (layer == 2) {
                    if (!offchip.read_addr.full()) {
                        offchip.read_addr.write(i);
                    }
                    if (!offchip.read_data.empty()) {
                        op1 = offchip.read_data.read(nullptr);
                    }
                // Use onchip for layer1/layer2
                } else {
                    op1 = (layer == 0) ? X[j] : scratchpad[j];
                }

                layer_inner_16: for (int jj = 0; jj < 16; jj++) {
                    #pragma HLS latency max=2
                    int16_v16 op2;

                    if (layer == 0) {
                        if (!W1.read_addr.full()) {
                            W1.read_addr.write(i*bound_j + j);
                        }
                        if (!W1.read_data.empty()) {
                            op2 = W1.read_data.read(nullptr);
                        }
                    } else if (layer == 1) {
                        if (!W2.read_addr.full()) {
                            W2.read_addr.write(i*bound_j + j);
                        }
                        if (!W2.read_data.empty()) {
                            op2 = W2.read_data.read(nullptr);
                        }
                    } else if (layer == 2) {
                        if (!W3.read_addr.full()) {
                            W3.read_addr.write(i*bound_j + j);
                        }
                        if (!W3.read_data.empty()) {
                            op2 = W3.read_data.read(nullptr);
                        }
                    }

                    layer_compute: for (int k = 0; k < 2; k++) {
                        #pragma HLS pipeline II=1
                        for (int kk = 0; kk < 8; kk++) {
                            #pragma HLS unroll
                            sum[k*8 + kk] = op1[jj] * op2[k*8 + kk];
                        }
                    }
                }
            }
            // Write to scratchpad for layer 1
            if (layer == 0) {
                scratchpad[i] = sum;
            }
            // Write offchip for layer 2
            else if (layer == 1) {
                if (!offchip.write_addr.full() && !offchip.write_data.full()) {
                    offchip.write_addr.write(i);
                    offchip.write_data.write(sum);
                }
                if (!offchip.write_resp.empty()) {
                    offchip.write_resp.read(nullptr);
                }
            // Write output for layer 3
            } else {
                if (!data_out.write_addr.full() && !data_out.write_data.full()) {
                    data_out.write_addr.write(i);
                    data_out.write_data.write(sum);
                }
                if (!data_out.write_resp.empty()) {
                    data_out.write_resp.read(nullptr);
                }
            }
        }
    }

    fifo_fin.write(true);
}

void MLP(
    tapa::mmap<int16_v16> X,
    tapa::mmap<int16_v16> W1,
    tapa::mmap<int16_v16> W2,
    tapa::mmap<int16_v16> W3,
    tapa::mmap<int16_v16> offchip,
    tapa::mmap<int16_v16> data_out,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(top, X, W1, W2, W3, offchip, data_out, fifo_fin)

        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count)
    ;
}
