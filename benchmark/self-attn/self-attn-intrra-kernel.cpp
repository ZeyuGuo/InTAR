#include <iostream>
#include <cmath>
#include <numeric>
#include <string>
#include <tapa.h>
#include <ap_int.h>

constexpr int seq_len = 256;
constexpr int D = 1024;
constexpr int D_head = 1024;
constexpr int D_head_div_2 = D_head / 2;
constexpr int weight_size_cc0 = D * D_head / 16;
constexpr int weight_size_cc1 = D * D_head / 8;
constexpr int input_size = seq_len * D / 16;
constexpr int output_size = seq_len * D_head / 32;
constexpr float scale = 1/32.0;

using float_v16 = tapa::vec_t<float, 16>;

struct ConfigInst {
    ap_uint<3> stage;
    ap_uint<8> i_bound;
    ap_uint<8> j_bound;
    ap_uint<8> k_bound;
};

// Kernel

void read_W(
    const int w_size,
    tapa::async_mmap<float_v16>& vec,
    tapa::ostream<float_v16>& fifo_out
){
    for(int i_req = 0, i_resp = 0; i_resp < w_size;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < w_size) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        float_v16 tmp_o; 
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

void read_X(
    tapa::async_mmap<float_v16>& vec,
    tapa::ostream<float_v16>& fifo_out
){
    for(int i_req = 0, i_resp = 0; i_resp < input_size;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < input_size) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        float_v16 tmp_o; 
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

void write_mtx(
    tapa::async_mmap<float_v16>& output_mtx,
    tapa::istream<float_v16>& fifo_in
){

    for(int i_req = 0, i_resp = 0; i_resp < output_size;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < output_size) & !fifo_in.empty() & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
            output_mtx.write_addr.try_write(i_req);
            float_v16 tmp; fifo_in.try_read(tmp);
            output_mtx.write_data.try_write(tmp);
            ++i_req;
        }
        bool success = false;
        auto resp = output_mtx.write_resp.read(success);
        if(success){
            i_resp += unsigned(resp)+1;
        }
    }
} 

void CC0_QAV_Proj(
    tapa::istream<float_v16>& fifo_input,
    tapa::istream<float_v16>& fifo_weight,
    // tapa::istream<ConfigInst>& fifo_inst,
    tapa::istream<float_v16>& fifo_from_CC1,
    tapa::ostream<float>& fifo_to_SFU,
    tapa::istream<float>& fifo_from_SFU,
    tapa::ostream<float_v16>& fifo_output
){
    // stage 1: compute Q
    float X[seq_len][D];
    float weight[D_head_div_2][D];
    float scratchpad[seq_len][D_head];

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < (D >> 4);){
            if(!fifo_input.empty()){
                float_v16 tmp; fifo_input.try_read(tmp);
                for(int k = 0; k < 16; k++){
                    X[i][j*16+k] = tmp[k];
                }
                j++;
            }
        }
    }

    for(int i = 0; i < D; i++){
        for(int j = 0; j < (D_head >> 5);){
            if(!fifo_weight.empty()){
                float_v16 tmp; fifo_weight.try_read(tmp);
                for(int k = 0; k < 16; k++){
                    weight[j*16+k][i] = tmp[k];
                }
                j++;
            }
        }
    }

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < (D_head >> 5); j++){
            float sum[16];
            for(int k = 0; k < 16; k++){
                sum[k] = 0.f;
            }
            for(int k = 0; k < D; k++){
                for(int l = 0; l < 16; l++){
                    sum[l] += X[i][k] * weight[j*16+l][k];
                }
            }
            float_v16 tmp_acc1 = fifo_from_CC1.read();
            for(int k = 0; k < 16; k++){
                scratchpad[i][j*32+k] = sum[k];
                scratchpad[i][j*32+k+16] = tmp_acc1[k];
            }
        }
    }

    LOG(INFO) << "fin Q";

    //stage 2: compute QK^T
    for(int i = 0; i < seq_len; i++){
        // load K
        float cache_k[D_head];
        for(int j = 0; j < (D_head >> 4);){
            if(!fifo_from_CC1.empty()){
                float_v16 tmp; fifo_from_CC1.try_read(tmp);
                for(int k = 0; k < 16; k++){
                    cache_k[k] = tmp[k];
                }
                j++;
            }
        }
        //compute
        for(int j = 0; j < seq_len; j++){
            float_v16 sum;
            for(int k = 0; k < 16; k++){
                sum[0] = 0.f;
            }
            for(int k = 0; k < (D_head >> 4); k++){
                for(int l = 0; l < 16; l++){
                    sum[l] += cache_k[k*16+l] * scratchpad[j][k*16+l];
                }
            }
            for(int k = 1; k < 16; k++){
                sum[0] += sum[k];
            }
            fifo_to_SFU.write(sum[0]);
        }
    }

    LOG(INFO) << "fin QK";

    // stage 3: compute V
    for(int i = 0; i < D; i++){
        for(int j = 0; j < (D_head >> 5);){
            if(!fifo_weight.empty()){
                float_v16 tmp; fifo_weight.try_read(tmp);
                for(int k = 0; k < 16; k++){
                    weight[j*16+k][i] = tmp[k];
                }
                j++;
            }
        }
    }

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < (D_head >> 5); j++){
            float sum[16];
            for(int k = 0; k < 16; k++){
                sum[k] = 0.f;
            }
            for(int k = 0; k < D; k++){
                for(int l = 0; l < 16; l++){
                    sum[l] += X[i][k] * weight[j*16+l][k];
                }
            }
            for(int k = 0; k < 16; k++){
                scratchpad[i][j*16+k] = sum[k];
            }
        }
    }

    LOG(INFO) << "fin V";

    // stage 4: Compute AV
    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < (D_head >> 5); j++){
            float_v16 sum;
            for(int k = 0; k < 16; k++){
                sum[0] = 0.f;
            }
            for(int k = 0; k < seq_len;){
                if(!fifo_from_SFU.empty()){
                    float tmp; fifo_from_SFU.try_read(tmp);
                    for(int l = 0; l < 16; l++){
                        sum[l] += tmp * scratchpad[k][j*16+l];
                    }
                    k++;
                }
            }
            fifo_output.write(sum);
        }
    }
}

void CC1_QKV_Proj(
    tapa::istream<float_v16>& fifo_input,
    tapa::istream<float_v16>& fifo_weight,
    // tapa::istream<ConfigInst>& fifo_inst,
    tapa::ostream<float_v16>& fifo_to_CC0,
    tapa::istream<float>& fifo_from_SFU,
    tapa::ostream<float_v16>& fifo_output
){
    // stage 1: compute Q
    float X[seq_len][D];
    float weight[D_head][D];
    float scratchpad[seq_len][D_head_div_2];

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < (D >> 4);){
            if(!fifo_input.empty()){
                float_v16 tmp; fifo_input.try_read(tmp);
                for(int k = 0; k < 16; k++){
                    X[i][j*16+k] = tmp[k];
                }
                j++;
            }
        }
    }

    for(int i = 0; i < D; i++){
        for(int j = 0; j < (D_head >> 5);){
            if(!fifo_weight.empty()){
                float_v16 tmp; fifo_weight.try_read(tmp);
                for(int k = 0; k < 16; k++){
                    weight[j*16+k][i] = tmp[k];
                }
                j++;
            }
        }
    }

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < (D_head >> 5); j++){
            float_v16 sum;
            for(int k = 0; k < 16; k++){
                sum[0] = 0.f;
            }
            for(int k = 0; k < D; k++){
                for(int l = 0; l < 16; l++){
                    sum[l] += X[i][k] * weight[j*16+l][k];
                }
            }
            fifo_to_CC0.write(sum);
        }
    }

    //stage 2: compute K
    for(int i = 0; i < D; i++){
        for(int j = 0; j < (D_head >> 4);){
            if(!fifo_weight.empty()){
                float_v16 tmp; fifo_weight.try_read(tmp);
                for(int k = 0; k < 16; k++){
                    weight[j*16+k][i] = tmp[k];
                }
                j++;
            }
        }
    }

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < (D_head >> 4); j++){
            float_v16 sum;
            for(int k = 0; k < 16; k++){
                sum[0] = 0.f;
            }
            for(int k = 0; k < D; k++){
                for(int l = 0; l < 16; l++){
                    sum[l] += X[i][k] * weight[j*16+l][k];
                }
            }
            fifo_to_CC0.write(sum);
        }
    }

    // stage 3: compute V
    for(int i = 0; i < D; i++){
        for(int j = 0; j < (D_head >> 5);){
            if(!fifo_weight.empty()){
                float_v16 tmp; fifo_weight.try_read(tmp);
                for(int k = 0; k < 16; k++){
                    weight[j*16+k][i] = tmp[k];
                }
                j++;
            }
        }
    }

    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < (D_head >> 5); j++){
            float sum[16];
            for(int k = 0; k < 16; k++){
                sum[k] = 0.f;
            }
            for(int k = 0; k < D; k++){
                for(int l = 0; l < 16; l++){
                    sum[l] += X[i][k] * weight[j*16+l][k];
                }
            }
            for(int k = 0; k < 16; k++){
                scratchpad[i][j*16+k] = sum[k];
            }
        }
    }

    LOG(INFO) << "fin V";

    // stage 4: Compute AV
    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < (D_head >> 5); j++){
            float_v16 sum;
            for(int k = 0; k < 16; k++){
                sum[0] = 0.f;
            }
            for(int k = 0; k < seq_len;){
                if(!fifo_from_SFU.empty()){
                    float tmp; fifo_from_SFU.try_read(tmp);
                    for(int l = 0; l < 16; l++){
                        sum[l] += tmp * scratchpad[k][j*16+l];
                    }
                    k++;
                }
            }
            fifo_output.write(sum);
        }
    }
}

void SFU_softmax(
    tapa::istream<float>& fifo_from_CC0,
    tapa::ostream<float>& fifo_to_CC0,
    tapa::ostream<float>& fifo_to_CC1
){
    float scaled_attn[seq_len][seq_len];
    
    for(int i = 0; i < seq_len; i++){
        float sum = 0.f;
        for(int j = 0; j < seq_len;){
            if(!fifo_from_CC0.empty()){
                float tmp; fifo_from_CC0.try_read(tmp);
                float score = std::exp(tmp * scale);
                sum += score;
                scaled_attn[i][j] = score;
                j++;
            }
        }
        sum = 1 / sum;
        for(int j = 0; j < seq_len; j++){
            scaled_attn[i][j] *= sum;
        }
    }

    LOG(INFO) << "fin softmax";

    // send back to CC
    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < (D_head >> 5); j++){
            for(int k = 0; k < seq_len; k++){
                fifo_to_CC0.write(scaled_attn[i][k]);
                fifo_to_CC1.write(scaled_attn[i][k]);
            }
        }
    }
}

// Self-attention computation
void selfAttention(
    tapa::mmap<float_v16> X_acc0,
    tapa::mmap<float_v16> X_acc1,
    tapa::mmap<float_v16> W_acc0,
    tapa::mmap<float_v16> W_acc1,
    tapa::mmap<float_v16> acc0_out,
    tapa::mmap<float_v16> acc1_out
) {
   
    tapa::stream<float_v16> fifo_input_CC0("fifo_input_CC0");
    tapa::stream<float_v16> fifo_input_CC1("fifo_input_CC1");
    tapa::stream<float_v16> fifo_weight_CC0("fifo_weight_CC0");
    tapa::stream<float_v16> fifo_weight_CC1("fifo_weight_CC1");

    tapa::stream<float> fifo_from_CC0_to_SFU("fifo_from_CC0_to_SFU");
    tapa::stream<float> fifo_from_SFU_to_CC0("fifo_from_SFU_to_CC0");
    tapa::stream<float> fifo_from_SFU_to_CC1("fifo_from_SFU_to_CC1");
    tapa::stream<float_v16> fifo_from_CC1_to_CC0("fifo_from_CC1_to_CC0");

    tapa::stream<float_v16> fifo_output_CC0("fifo_output_CC0");
    tapa::stream<float_v16> fifo_output_CC1("fifo_output_CC1");

    tapa::task()
        .invoke<tapa::join>(read_W, weight_size_cc0, W_acc0, fifo_weight_CC0)
        .invoke<tapa::join>(read_W, weight_size_cc1, W_acc1, fifo_weight_CC1)
        .invoke<tapa::join>(read_X, X_acc0, fifo_input_CC0)
        .invoke<tapa::join>(read_X, X_acc1, fifo_input_CC1)
        .invoke<tapa::join>(CC0_QAV_Proj, fifo_input_CC0, fifo_weight_CC0, fifo_from_CC1_to_CC0, fifo_from_CC0_to_SFU, fifo_from_SFU_to_CC0, fifo_output_CC0)
        .invoke<tapa::join>(SFU_softmax, fifo_from_CC0_to_SFU, fifo_from_SFU_to_CC0, fifo_from_SFU_to_CC1)
        .invoke<tapa::join>(CC1_QKV_Proj, fifo_input_CC1, fifo_weight_CC1, fifo_from_CC1_to_CC0, fifo_from_SFU_to_CC1, fifo_output_CC1)
        .invoke<tapa::join>(write_mtx, acc0_out, fifo_output_CC0)
        .invoke<tapa::join>(write_mtx, acc1_out, fifo_output_CC1);
}