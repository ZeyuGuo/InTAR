#include <cmath>
#include <string>
#include <tapa.h>
#include <ap_int.h>
#include <hls_math.h>

constexpr int D = 1024;
constexpr int D_div_2 = D / 2;
constexpr int D_div_4 = D / 4;
constexpr int D_ffn = 4096;
constexpr int N_head = 16;
constexpr int MAX_SEQ_LEN = 1024;
constexpr int MAX_SEQ_LEN_div_2 = MAX_SEQ_LEN / 2;
constexpr int MAX_SEQ_LEN_div_8 = MAX_SEQ_LEN / 8;
constexpr int NUM_SLR = 4;
constexpr int NUM_DUM_SLR = 4;
constexpr int TOTAL_PORT = NUM_SLR * 2;
constexpr int D_head = D / N_head;
constexpr int D_head_div_16 = D_head / 16;
constexpr int D_head_div_8 = D_head / 8;
constexpr int D_head_div_4 = D_head / 4;
constexpr int D_head_div_2 = D_head / 2;
constexpr int D_div_8 = D / 8;
constexpr int D_div_16 = D / 16;
constexpr int FFN_WEIGHT_SIZE = D * D_ffn;
constexpr int OUT_WEIGHT_SIZE = D * D_head * NUM_DUM_SLR * 4;
constexpr int WEIGHT_D = D * 2;
constexpr int QKV_WEIGHT_SIZE = D * D_head * NUM_DUM_SLR * 8; // multi-head attention
constexpr int TOTAL_WEIGHT_SIZE = OUT_WEIGHT_SIZE + QKV_WEIGHT_SIZE;
constexpr int WEIGHT_SIZE = TOTAL_WEIGHT_SIZE + FFN_WEIGHT_SIZE;
constexpr int CONTEXT_D = D_head_div_8 * 4;
constexpr int D_head_mul_4 = D_head * 4;
constexpr int D_write_zero = D / 32;

using int_v16 = tapa::vec_t<int, 16>;
using int4_v128 = tapa::vec_t<ap_int<4>, 128>;
using int8_v64 = tapa::vec_t<ap_int<8>, 64>;

template <typename data_t>
inline void bh(tapa::istream<data_t> & q) {
#pragma HLS inline
    for (;;) {
#pragma HLS pipeline II=1 style=stp style=stp
        data_t tmp; q.try_read(tmp);
    }
}

void black_hole_int(tapa::istream<int> & fifo_in) {
    bh(fifo_in);
}

void black_hole_int_v16(tapa::istream<int_v16> & fifo_in) {
    bh(fifo_in);
}

void black_hole_x(tapa::istream<int8_v64> & fifo_in) {
    bh(fifo_in);
}

void black_hole_w(tapa::istream<int4_v128> & fifo_in) {
    bh(fifo_in);
}

void black_hole_ap_uint_512(tapa::istream<ap_uint<512>> & fifo_in) {
    bh(fifo_in);
}

void black_hole_ap_uint_1024(tapa::istream<ap_uint<1024>> & fifo_in) {
    bh(fifo_in);
}

void read_W(
    const int N,
    tapa::async_mmap<ap_uint<512>>& vec,
    tapa::ostream<ap_uint<512>>& fifo_out
){

    for(int i_req = 0, i_resp = 0; i_resp < (N >> 7);){
        #pragma HLS pipeline II=1 style=stp style=stp
        if((i_req < (N >> 7)) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        ap_uint<512> tmp_o; 
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

void read_X(
    const int N,
    tapa::async_mmap<ap_uint<512>>& vec,
    tapa::ostream<ap_uint<512>>& fifo_out
){
    for(int i_req = 0, i_resp = 0; i_resp < (N >> 6);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < (N >> 6)) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        ap_uint<512> tmp_o; 
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

void read_inst(
    const int L,
    tapa::ostream<int>& fifo_out_acc0,
    tapa::ostream<int>& fifo_out_acc1
){
    for(int stage_i = 0; stage_i < 14; stage_i++){
        #pragma HLS pipeline II=1 style=stp

        const int stage = (stage_i < 12) ? (stage_i % 3) : (stage_i - 9);

        if(stage == 3){
            fifo_out_acc0.write(0);
            fifo_out_acc1.write(0);

            fifo_out_acc0.write(L/2);
            fifo_out_acc1.write(L/2);
        }
        else if(stage != 1){
            fifo_out_acc0.write(0);
            if(stage != 4) fifo_out_acc1.write(0);

            fifo_out_acc0.write(L);
            if(stage != 4) fifo_out_acc1.write(L);
        } else {
            fifo_out_acc0.write(0);
            fifo_out_acc0.write(L/2);

            fifo_out_acc1.write(L/2);
            fifo_out_acc1.write(L);
        } 
    }
}

void write_mtx(
    const int N,
    tapa::async_mmap<ap_uint<512>>& output_mtx,
    tapa::istream<ap_uint<512>>& fifo_in,
    tapa::ostream<bool>& fifo_fin
){

    for(int i_req = 0, i_resp = 0; i_resp < N;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < N) & !fifo_in.empty() & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
            output_mtx.write_addr.try_write(i_req);
            ap_uint<512> tmp; fifo_in.try_read(tmp);
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

void write_zero(
    const int L,
    tapa::ostream<ap_uint<512>>& fifo_zero
){
    for(int i = 0; i < L * D_write_zero;){
        if(!fifo_zero.full()){
            ap_uint<512> tmp = 0;
            fifo_zero.try_write(tmp);
            i++;
        }
    }
}

// acc slr0 master node
void temporal_acc0_slr0(
    const int L,
    tapa::istream<int>& fifo_len_in,
    tapa::ostream<int>& fifo_len_out,
    tapa::istream<ap_uint<512>>& fifo_X_in,
    tapa::ostream<ap_uint<1024>>& fifo_X_out, // 8-bit activation
    tapa::istream<ap_uint<512>>& fifo_W_in,
    tapa::ostream<ap_uint<512>>& fifo_W_out, // 4-bit weight
    tapa::istream<ap_uint<128>>& fifo_from_acc1,
    tapa::ostream<ap_uint<512>>& fifo_O_out,
    tapa::ostream<ap_uint<512>>& fifo_ffn_out,
    tapa::istream<ap_uint<1024>>& fifo_context,
    tapa::istream<ap_uint<1024>>& fifo_ffn_in,
    tapa::istream<ap_uint<512>>& fifo_reduce_recv,
    tapa::ostream<ap_uint<512>>& fifo_res_send
    // tapa::ostream<ap_uint<64>>& fifo_write,
    // tapa::ostream<bool>& fifo_fin
){

    ap_uint<64> scratchpad_q[MAX_SEQ_LEN][D_head_div_8]; // 8 bit
    #pragma HLS array_partition variable=scratchpad_q cyclic dim=1 factor=16
    #pragma HLS array_partition variable=scratchpad_q cyclic dim=2 factor=2

    ap_uint<64> scratchpad_k[MAX_SEQ_LEN][D_head_div_8]; // 8 bit
    #pragma HLS array_partition variable=scratchpad_k cyclic dim=1 factor=16
    #pragma HLS array_partition variable=scratchpad_k cyclic dim=2 factor=2
    #pragma HLS bind_storage variable=scratchpad_k type=ram_2p impl=uram 

    ap_uint<64> X[MAX_SEQ_LEN][D_div_8]; // 8 bit
    #pragma HLS array_partition variable=X cyclic dim=1 factor=16
    #pragma HLS array_partition variable=X cyclic dim=2 factor=2
    #pragma HLS bind_storage variable=X type=ram_2p impl=uram 

    for(int stage_i = 0; stage_i < 14; stage_i++){

        //TODO: stage send from inst

        // stage 0: WqX
        // stage 1: WkX0 <- acc1
        // stage 2: QK^T

        ap_uint<32> W[D][D_div_8]; // TODO: reduce dimension
        #pragma HLS array_partition variable=W cyclic dim=1 factor=16

        const int start = fifo_len_in.read();
        const int end = fifo_len_in.read();
        fifo_len_out.write(start);
        fifo_len_out.write(end);

        const int stage = (stage_i < 12) ? (stage_i % 3) : (stage_i - 9);

        // load weights and forward
        if(stage != 2) { // TODO: 1d array & uniform access
            int bound = (stage == 3) ? D_head : D_head_div_4;
            bound = (stage == 4) ? D_div_4 : bound;
            for(int i = 0; i < bound; i++){
                load_weight:
                for(int j = 0; j < D_div_8;){
                    if(!fifo_W_in.empty()){
                        ap_uint<512> val; fifo_W_in.try_read(val);

                        for(int k = 0; k < 4; k++){
                            #pragma HLS unroll
                            W[i*4+k][j] = ap_uint<32>(val(k*32+31, k*32));
                        }
                        val = ap_uint<512>(val >> 128);
                        fifo_W_out.write(val);
                        j++;
                    }
                }
            }
        }

        int j_bound = (stage == 2) ? (L >> 4) : D_head_div_16;
        j_bound = (stage >= 3) ? D_div_16 : j_bound;
        int k_bound = (stage > 1 && stage < 4) ? D_head_div_8 : D_div_8;
        k_bound = (stage == 3) ? D_head_div_2 : k_bound;
        
        // stage 1: compute Q 
        for(int i = (start >> 4); i < (end >> 4); i++){ // make sure L is multiple of 16

            if(stage_i == 0){
                for(int ii = 0; ii < 2; ii++){ // load only 1 time
        load_x:
                    for(int jj = 0; jj < D_div_8;){
                        if(!fifo_X_in.empty()){
                            ap_uint<512> val; fifo_X_in.try_read(val);
                            
                            for(int k = 0; k < 8; k++){
                                #pragma HLS unroll
                                X[i*16+ii*8+k][jj] = ap_uint<64>(val(k*64+63, k*64));
                            }
                            jj++;
                        }
                    }
                }
            }

            for(int j = 0; j < j_bound; j++){
                #pragma HLS loop_flatten off

                ap_int<22> acc_vec[2][3][16][16];
                #pragma HLS array_partition variable=acc_vec dim=1 complete
                #pragma HLS array_partition variable=acc_vec dim=2 complete
                #pragma HLS array_partition variable=acc_vec dim=3 complete
                #pragma HLS array_partition variable=acc_vec dim=4 complete

                for(int l = 0; l < 2; l++){
                    #pragma HLS unroll
                    for(int ii = 0; ii < 3; ii++){
                        #pragma HLS unroll
                        for(int kk = 0; kk < 16; kk++){
                            #pragma HLS unroll
                            for(int k = 0; k < 16; k++){
                                #pragma HLS unroll
                                acc_vec[l][ii][kk][k] = 0;
                            }
                        }
                    }
                }

        compute:
                for(int k = 0; k < k_bound; k++){ // reduction dim
                    #pragma HLS pipeline II=1 style=stp
                    #pragma HLS dependence variable=acc_vec type=intra direction=RAW false
                    #pragma HLS dependence variable=acc_vec type=intra direction=WAR true

                    ap_uint<72> op1_mtx[16];
                    ap_uint<72> op2_mtx[16];
                    #pragma HLS array_partition variable=op1_mtx complete
                    #pragma HLS array_partition variable=op2_mtx complete

                    ap_uint<1024> recv_pkt;

                    if(stage == 3) {
                        recv_pkt = fifo_context.read();
                    }else if(stage == 4) {
                        recv_pkt = fifo_ffn_in.read();
                    }

                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS unroll
                        if(stage > 2){
                            op1_mtx[ii] = ap_uint<72>(ap_uint<36>((ap_int<4>(2), W[k*8+ii%8][j*2+ii/8]))); // dummy to enforce dotpra3 binding
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), recv_pkt(ii*64+63, ii*64)));
                        } else if(stage == 2) {
                            op1_mtx[ii] = ap_uint<72>((ap_int<8>(2), scratchpad_q[i*16+ii][k]));
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), scratchpad_k[j*16+ii][k]));
                        } else {
                            op1_mtx[ii] = ap_uint<72>(ap_uint<36>((ap_int<4>(2), W[j*16+ii][k])));
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), X[i*16+ii][k]));
                        }
                    }

                    if(stage < 2){
                        ap_uint<1024> send_pkt = ap_uint<1024>((
                            op2_mtx[0](63,0), op2_mtx[1](63,0), op2_mtx[2](63,0), op2_mtx[3](63,0), op2_mtx[4](63,0), op2_mtx[5](63,0), op2_mtx[6](63,0), op2_mtx[7](63,0),
                            op2_mtx[8](63,0), op2_mtx[9](63,0), op2_mtx[10](63,0), op2_mtx[11](63,0), op2_mtx[12](63,0), op2_mtx[13](63,0), op2_mtx[14](63,0), op2_mtx[15](63,0)
                        ));
                        fifo_X_out.write(send_pkt);
                    } else if (stage == 4) {
                        fifo_X_out.write(recv_pkt);
                    }

                    for(int kk = 0; kk < 16; kk++){
                        #pragma HLS unroll
                        for(int l = 0; l < 16; l++){
                            #pragma HLS unroll
                            ap_int<22> res0 = acc_vec[k%2][0][kk][l];
                            ap_int<22> res1 = acc_vec[k%2][1][kk][l];
                            ap_int<22> res2 = acc_vec[k%2][2][kk][l];
                            ap_int<8> a0; ap_int<8> a1; ap_int<8> a2;
                            ap_int<8> b0; ap_int<8> b1; ap_int<8> b2;
                            ap_int<8> a0_1; ap_int<8> a1_1; ap_int<8> a2_1;
                            ap_int<8> b0_1; ap_int<8> b1_1; ap_int<8> b2_1;
                            ap_int<8> a0_2; ap_int<8> a1_2; ap_int<8> a2_2;
                            ap_int<8> b0_2; ap_int<8> b1_2; ap_int<8> b2_2;
                            (a2_2, a1_2, a0_2, a2_1, a1_1, a0_1, a2, a1, a0) = op1_mtx[l];
                            if(stage==2){
                                (b2_2, b1_2, b0_2, b2_1, b1_1, b0_1, b2, b1, b0) = op2_mtx[kk];
                            }else{
                                b0 = ap_int<4>(op2_mtx[kk](3, 0));
                                b1 = ap_int<4>(op2_mtx[kk](7, 4));
                                b2 = ap_int<4>(op2_mtx[kk](11, 8));
                                b0_1 = ap_int<4>(op2_mtx[kk](15, 12));
                                b1_1 = ap_int<4>(op2_mtx[kk](19, 16));
                                b2_1 = ap_int<4>(op2_mtx[kk](23, 20));
                                b0_2 = ap_int<4>(op2_mtx[kk](27, 24));
                                b1_2 = ap_int<4>(op2_mtx[kk](31, 28));
                                b2_2 = ap_int<4>(op2_mtx[kk](35, 32));
                            }
                            res0 = (((a0 * b0) + (a1 * b1)) + (a2 * b2)) + res0;
                            res1 = (((a0_1 * b0_1) + (a1_1 * b1_1)) + (a2_1 * b2_1)) + res1;
                            res2 = (((a0_2 * b0_2) + (a1_2 * b1_2)) + (a2_2 * b2_2)) + res2;
                            acc_vec[k%2][0][kk][l] = res0;
                            acc_vec[k%2][1][kk][l] = res1;
                            acc_vec[k%2][2][kk][l] = res2;
                        }
                    }
                }

                ap_int<24> acc_final[16][16];
                #pragma HLS array_partition variable=acc_final dim=1 complete
                #pragma HLS array_partition variable=acc_final dim=2 complete

                for(int ii = 0; ii < 16; ii++){
                    #pragma HLS unroll
                    for(int k = 0; k < 16; k++){
                        #pragma HLS unroll
                        acc_final[ii][k] = -(k_bound << 2);
                    }
                }

        reduction:
                for(int k = 0; k < 2; k++){
                    for(int ii = 0; ii < 3; ii++){
                        for(int kk = 0; kk < 16; kk++){
                            #pragma HLS unroll
                            for(int l = 0; l < 16; l++){
                                #pragma HLS unroll
                                acc_final[kk][l] += acc_vec[k][ii][kk][l];
                            }
                        }
                    }
                }

               if(stage == 0){
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS unroll
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            int offset = k%8;
                            scratchpad_q[i*16+ii][j*2+k/8](offset*8+7, offset*8) = ap_int<8>(acc_final[ii][k] >> 5);
                        }
                    }
                } else if (stage == 1){
                    for(int ii = 0; ii < 16; ii++){
                        ap_uint<128> tmp = fifo_from_acc1.read();
                        
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            int offset = k%8;
                            scratchpad_k[i*16+ii][j*2+k/8](offset*8+7, offset*8) = ap_int<8>(acc_final[ii][k] >> 5);
                        }
                        for(int k = 0; k < 2; k++){
                            #pragma HLS unroll
                            scratchpad_k[end + i*16 + ii][j*2+k] = ap_uint<64>(tmp(k*64+63, k*64));
                        }
                    }
                } else if(stage == 2 || stage == 4){
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS pipeline II=1 style=stp
                        ap_uint<512> tmp;
                        for(int kk = 0; kk < 16; kk++){
                            #pragma HLS unroll
                            tmp(kk*32+31, kk*32) = tapa::bit_cast<ap_uint<32>>(acc_final[ii][kk]);
                        }
                        if(stage == 2) fifo_O_out.write(tmp);
                        else fifo_ffn_out.write(tmp);
                    }
                } else {
                final_acc:
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS pipeline II=1 style=stp
                        #pragma HLS dependence variable=X type=inter false
                        ap_uint<512> tmp_recv = fifo_reduce_recv.read();
                        ap_uint<512> tmp_send;
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            ap_int<32> tmp = acc_final[ii][k] + ap_int<22>(tmp_recv(k*32+21, k*32));
                            tmp += ap_int<8>(X[i*16+ii][j*2+k/8]((k%8)*8+7, (k%8)*8));
                            tmp_send(k*32+31, k*32) = tmp;
                        }
                        fifo_res_send.write(tmp_send);
                    }
                }
            }
        }
    }
    // fifo_fin.write(true);

    // write:
    // for(int i = 0; i < L; i++){
    //     for(int j = 0; j < D_div_8; j++){
    //         #pragma HLS pipeline II=1 style=stp
    //         fifo_write.write(X[i][j]);
    //     }
    // }
}

void temporal_acc0(
    const int L,
    tapa::istream<int>& fifo_len_in,
    tapa::ostream<int>& fifo_len_out,
    tapa::istream<ap_uint<1024>>& fifo_X_in,
    tapa::ostream<ap_uint<1024>>& fifo_X_out, // 8-bit activation
    tapa::istream<ap_uint<512>>& fifo_W_in,
    tapa::ostream<ap_uint<512>>& fifo_W_out, // 4-bit weight
    tapa::istream<ap_uint<128>>& fifo_from_acc1,
    tapa::ostream<ap_uint<512>>& fifo_O_out,
    tapa::istream<ap_uint<1024>>& fifo_context,
    tapa::ostream<ap_uint<512>>& fifo_ffn_out,
    tapa::istream<ap_uint<512>>& fifo_reduce_recv,
    tapa::ostream<ap_uint<512>>& fifo_reduce_send
){

    ap_uint<64> scratchpad_q[MAX_SEQ_LEN][D_head_div_8]; // 8 bit
    #pragma HLS array_partition variable=scratchpad_q cyclic dim=1 factor=16
    #pragma HLS array_partition variable=scratchpad_q cyclic dim=2 factor=2

    ap_uint<64> scratchpad_k[MAX_SEQ_LEN][D_head_div_8]; // 8 bit
    #pragma HLS array_partition variable=scratchpad_k cyclic dim=1 factor=16
    #pragma HLS array_partition variable=scratchpad_k cyclic dim=2 factor=2
    #pragma HLS bind_storage variable=scratchpad_k type=ram_2p impl=uram

    for(int stage_i = 0; stage_i < 14; stage_i++){
    #pragma HLS loop_flatten off

        // stage 0: WqX
        // stage 1: WkX0 <- acc1
        // stage 2: QK^T
        // stage 3: WoO

        ap_uint<32> W[D][D_div_8]; // 4 bit
        #pragma HLS array_partition variable=W cyclic dim=1 factor=16

        const int start = fifo_len_in.read();
        const int end = fifo_len_in.read();
        fifo_len_out.write(start);
        fifo_len_out.write(end);

        const int stage = (stage_i < 12) ? (stage_i % 3) : stage_i - 9;

        // load weights and forward
        if(stage != 2) {
            int bound = (stage == 3) ? D_head : D_head_div_4;
            bound = (stage == 4) ? D_div_4 : bound;
            for(int i = 0; i < bound; i++){
                load_weight:
                for(int j = 0; j < D_div_8;){
                    if(!fifo_W_in.empty()){
                        ap_uint<512> val; fifo_W_in.try_read(val);

                        for(int k = 0; k < 4; k++){
                            #pragma HLS unroll
                            W[i*4+k][j] = ap_uint<32>(val(k*32+31, k*32));
                        }
                        val = ap_uint<512>(val >> 128);
                        fifo_W_out.write(val);
                        j++;
                    }
                }
            }
        }

        int j_bound = (stage == 2) ? (L >> 4) : D_head_div_16;
        j_bound = (stage >= 3) ? D_div_16 : j_bound;
        int k_bound = (stage > 1 && stage < 4) ? D_head_div_8 : D_div_8;
        k_bound = (stage == 3) ? D_head_div_2 : k_bound;
        
        // stage 1: compute Q 
        for(int i = (start >> 4); i < (end >> 4); i++){ // make sure L is multiple of 64
            for(int j = 0; j < j_bound; j++){
                #pragma HLS loop_flatten off

                ap_int<22> acc_vec[2][3][16][16];
                #pragma HLS array_partition variable=acc_vec dim=1 complete
                #pragma HLS array_partition variable=acc_vec dim=2 complete
                #pragma HLS array_partition variable=acc_vec dim=3 complete
                #pragma HLS array_partition variable=acc_vec dim=4 complete

                for(int l = 0; l < 2; l++){
                    #pragma HLS unroll
                    for(int ii = 0; ii < 3; ii++){
                        #pragma HLS unroll
                        for(int kk = 0; kk < 16; kk++){
                            #pragma HLS unroll
                            for(int k = 0; k < 16; k++){
                                #pragma HLS unroll
                                acc_vec[l][ii][kk][k] = 0;
                            }
                        }
                    }
                }

        compute:
                for(int k = 0; k < k_bound; k++){ // reduction dim
                    #pragma HLS pipeline II=1 style=stp
                    #pragma HLS dependence variable=acc_vec type=intra direction=RAW false
                    #pragma HLS dependence variable=acc_vec type=intra direction=WAR true

                    ap_uint<72> op1_mtx[16];
                    ap_uint<72> op2_mtx[16];
                    #pragma HLS array_partition variable=op1_mtx complete
                    #pragma HLS array_partition variable=op2_mtx complete

                    ap_uint<1024> recv_pkt;
                    if(stage == 3){
                        recv_pkt = fifo_context.read();
                    } else if(stage != 2) {
                        recv_pkt = fifo_X_in.read();
                        fifo_X_out.write(recv_pkt);
                    }

                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS unroll
                        if(stage > 2){
                            op1_mtx[ii] = ap_uint<72>(ap_uint<36>((ap_int<4>(2), W[k*8+ii%8][j*2+ii/8]))); // dummy to enforce dotpra3 binding
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), recv_pkt(ii*64+63, ii*64)));
                        } else if(stage == 2) {
                            op1_mtx[ii] = ap_uint<72>((ap_int<8>(2), scratchpad_q[i*16+ii][k]));
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), scratchpad_k[j*16+ii][k]));
                        } else {
                            op1_mtx[ii] = ap_uint<72>(ap_uint<36>((ap_int<4>(2), W[j*16+ii][k])));
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), recv_pkt(ii*64+63, ii*64)));
                        }
                    }

                    for(int kk = 0; kk < 16; kk++){
                        #pragma HLS unroll
                        for(int l = 0; l < 16; l++){
                            #pragma HLS unroll
                            ap_int<22> res0 = acc_vec[k%2][0][kk][l];
                            ap_int<22> res1 = acc_vec[k%2][1][kk][l];
                            ap_int<22> res2 = acc_vec[k%2][2][kk][l];
                            ap_int<8> a0; ap_int<8> a1; ap_int<8> a2;
                            ap_int<8> b0; ap_int<8> b1; ap_int<8> b2;
                            ap_int<8> a0_1; ap_int<8> a1_1; ap_int<8> a2_1;
                            ap_int<8> b0_1; ap_int<8> b1_1; ap_int<8> b2_1;
                            ap_int<8> a0_2; ap_int<8> a1_2; ap_int<8> a2_2;
                            ap_int<8> b0_2; ap_int<8> b1_2; ap_int<8> b2_2;
                            (a2_2, a1_2, a0_2, a2_1, a1_1, a0_1, a2, a1, a0) = op1_mtx[l];
                            if(stage==2){
                                (b2_2, b1_2, b0_2, b2_1, b1_1, b0_1, b2, b1, b0) = op2_mtx[kk];
                            }else{
                                b0 = ap_int<4>(op2_mtx[kk](3, 0));
                                b1 = ap_int<4>(op2_mtx[kk](7, 4));
                                b2 = ap_int<4>(op2_mtx[kk](11, 8));
                                b0_1 = ap_int<4>(op2_mtx[kk](15, 12));
                                b1_1 = ap_int<4>(op2_mtx[kk](19, 16));
                                b2_1 = ap_int<4>(op2_mtx[kk](23, 20));
                                b0_2 = ap_int<4>(op2_mtx[kk](27, 24));
                                b1_2 = ap_int<4>(op2_mtx[kk](31, 28));
                                b2_2 = ap_int<4>(op2_mtx[kk](35, 32));
                            }
                            res0 = (((a0 * b0) + (a1 * b1)) + (a2 * b2)) + res0;
                            res1 = (((a0_1 * b0_1) + (a1_1 * b1_1)) + (a2_1 * b2_1)) + res1;
                            res2 = (((a0_2 * b0_2) + (a1_2 * b1_2)) + (a2_2 * b2_2)) + res2;
                            acc_vec[k%2][0][kk][l] = res0;
                            acc_vec[k%2][1][kk][l] = res1;
                            acc_vec[k%2][2][kk][l] = res2;
                        }
                    }
                }

                ap_int<24> acc_final[16][16];
                #pragma HLS array_partition variable=acc_final dim=1 complete
                #pragma HLS array_partition variable=acc_final dim=2 complete

                for(int ii = 0; ii < 16; ii++){
                    #pragma HLS unroll
                    for(int k = 0; k < 16; k++){
                        #pragma HLS unroll
                        acc_final[ii][k] = -(k_bound << 2);
                    }
                }

        reduction:
                for(int k = 0; k < 2; k++){
                    for(int ii = 0; ii < 3; ii++){
                        for(int kk = 0; kk < 16; kk++){
                            #pragma HLS unroll
                            for(int l = 0; l < 16; l++){
                                #pragma HLS unroll
                                acc_final[kk][l] += acc_vec[k][ii][kk][l];
                            }
                        }
                    }
                }

                if(stage == 0){
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS unroll
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            int offset = k%8;
                            scratchpad_q[i*16+ii][j*2+k/8](offset*8+7, offset*8) = ap_int<8>(acc_final[ii][k] >> 5);
                        }
                    }
                } else if (stage == 1){
                    for(int ii = 0; ii < 16; ii++){
                        ap_uint<128> tmp = fifo_from_acc1.read();
                        
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            int offset = k%8;
                            scratchpad_k[i*16+ii][j*2+k/8](offset*8+7, offset*8) = ap_int<8>(acc_final[ii][k] >> 5);
                        }
                        for(int k = 0; k < 2; k++){
                            #pragma HLS unroll
                            scratchpad_k[end + i*16 + ii][j*2+k] = ap_uint<64>(tmp(k*64+63, k*64));
                        }
                    }
                } else if(stage == 2 || stage == 4){
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS pipeline II=1 style=stp
                        ap_uint<512> tmp;
                        for(int kk = 0; kk < 16; kk++){
                            #pragma HLS unroll
                            tmp(kk*32+31, kk*32) = tapa::bit_cast<ap_uint<32>>(acc_final[ii][kk]);
                        }
                        if(stage == 2) fifo_O_out.write(tmp);
                        else fifo_ffn_out.write(tmp);
                    }
                } else {
                    final_acc:
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS pipeline II=1 style=stp
                        ap_uint<512> tmp_recv = fifo_reduce_recv.read();
                        ap_uint<512> tmp;
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            acc_final[ii][k] += ap_int<22>(tmp_recv(k*32+21, k*32));
                            tmp(k*32+21, k*32) = acc_final[ii][k];
                        }
                        fifo_reduce_send.write(tmp);
                    }
                }
            }
        }
    }
}

// acc slr0 master node
void temporal_acc1_slr0(
    const int L,
    tapa::istream<int>& fifo_len_in,
    tapa::ostream<int>& fifo_len_out,
    tapa::istream<ap_uint<512>>& fifo_X_in,
    tapa::ostream<ap_uint<1024>>& fifo_X_out, // 8-bit activation
    tapa::istream<ap_uint<512>>& fifo_W_in,
    tapa::ostream<ap_uint<512>>& fifo_W_out, // 4-bit weight
    tapa::ostream<ap_uint<128>>& fifo_to_acc0,
    tapa::istream<ap_uint<128>>& fifo_from_sfu,
    tapa::ostream<ap_uint<1024>>& fifo_O_out,
    tapa::istream<ap_uint<1024>>& fifo_context,
    tapa::istream<ap_uint<512>>& fifo_reduce_recv,
    tapa::ostream<ap_uint<512>>& fifo_res_send
    // tapa::ostream<ap_uint<64>>& fifo_write,
    // tapa::ostream<bool>& fifo_fin
){
    ap_uint<64> X[MAX_SEQ_LEN][D_div_8]; // 8 bit
    #pragma HLS array_partition variable=X cyclic dim=1 factor=16
    #pragma HLS array_partition variable=X cyclic dim=2 factor=2
    #pragma HLS bind_storage variable=X type=ram_2p impl=uram

    ap_uint<64> scratchpad[MAX_SEQ_LEN][D_head_div_8]; // 8 bit
    #pragma HLS array_partition variable=scratchpad cyclic dim=1 factor=16
    #pragma HLS array_partition variable=scratchpad cyclic dim=2 factor=2
    #pragma HLS bind_storage variable=scratchpad type=ram_2p impl=uram

    // ap_uint<64> scratchpad_out[MAX_SEQ_LEN][D_head_div_8];
    // #pragma HLS array_partition variable=scratchpad_out cyclic dim=1 factor=16
    // #pragma HLS array_partition variable=scratchpad_out cyclic dim=2 factor=2

    for(int stage_i = 0; stage_i < 13; stage_i++){

        // stage 0: WvX
        // stage 1: WkX1 -> acc0
        // stage 2: Softmax(QK)V <- acc0
        // stage 3: WoO

        ap_uint<32> W[D_head_mul_4][D_div_8]; // 4 bit
        #pragma HLS array_partition variable=W cyclic dim=1 factor=16

        const int start = fifo_len_in.read();
        const int end = fifo_len_in.read();
        fifo_len_out.write(start);
        fifo_len_out.write(end);

        const int stage = (stage_i < 12) ? (stage_i % 3) : 3;
        // load weights and forward
        if(stage != 2) {
            const int bound = (stage == 3) ? D_head : D_head_div_4;
            for(int i = 0; i < bound; i++){
                load_weight:
                for(int j = 0; j < D_div_8;){
                    if(!fifo_W_in.empty()){
                        ap_uint<512> val; fifo_W_in.try_read(val);

                        for(int k = 0; k < 4; k++){
                            #pragma HLS unroll
                            W[i*4+k][j] = ap_uint<32>(val(k*32+31, k*32));
                        }
                        val = ap_uint<512>(val >> 128);
                        fifo_W_out.write(val);
                        j++;
                    }
                }
            }
        }

        int k_bound = (stage == 2) ? (L >> 3) : D_div_8;
        k_bound = (stage == 3) ? D_head_div_2 : k_bound;
        int j_bound = (stage == 3) ? D_div_16 : D_head_div_16;
        
        for(int i = (start >> 4); i < (end >> 4); i++){ // make sure L is multiple of 4

            ap_uint<32> cache_attn[MAX_SEQ_LEN_div_8][16];
            #pragma HLS array_partition variable=cache_attn dim=2 complete
            #pragma HLS array_partition variable=cache_attn dim=1 cyclic factor=2

            if(stage_i == 0){
                for(int ii = 0; ii < 2; ii++){ // load only 1 time
        load_x:
                    for(int jj = 0; jj < D_div_8;){
                        if(!fifo_X_in.empty()){
                            ap_uint<512> val; fifo_X_in.try_read(val);
                            
                            for(int k = 0; k < 8; k++){
                                #pragma HLS unroll
                                X[i*16+ii*8+k][jj] = ap_uint<64>(val(k*64+63, k*64));
                            }
                            jj++;
                        }
                    }
                }
            } else if (stage == 2) {
                for(int ii = 0; ii < (L >> 3); ii++){
                    ap_uint<32> fuse_reg[16];
                    load_attn:
                    for(int offset = 0; offset < 8;){
                        #pragma HLS pipeline II=1 style=stp
                        if(!fifo_from_sfu.empty()){
                            ap_uint<128> val; fifo_from_sfu.try_read(val);
                            for(int k = 0; k < 16; k++){
                                #pragma HLS unroll
                                fuse_reg[k](offset*4+3, offset*4) = ap_int<4>(val(k*8+3, k*8));
                            }
                            offset++;
                        }
                    }
                    for(int k = 0; k < 16; k++){
                        #pragma HLS unroll
                        cache_attn[ii][k] = fuse_reg[k];
                    }
                }
            }

            for(int j = 0; j < j_bound; j++){
                #pragma HLS loop_flatten off

                ap_int<22> acc_vec[2][3][16][16];
                #pragma HLS array_partition variable=acc_vec dim=1 complete
                #pragma HLS array_partition variable=acc_vec dim=2 complete
                #pragma HLS array_partition variable=acc_vec dim=3 complete
                #pragma HLS array_partition variable=acc_vec dim=4 complete

                for(int l = 0; l < 2; l++){
                    #pragma HLS unroll
                    for(int ii = 0; ii < 3; ii++){
                        #pragma HLS unroll
                        for(int kk = 0; kk < 16; kk++){
                            #pragma HLS unroll
                            for(int k = 0; k < 16; k++){
                                #pragma HLS unroll
                                acc_vec[l][ii][kk][k] = 0;
                            }
                        }
                    }
                }

        compute:
                for(int k = 0; k < k_bound; k++){
                    #pragma HLS pipeline II=1 style=stp
                    #pragma HLS dependence variable=acc_vec type=intra direction=RAW false
                    #pragma HLS dependence variable=acc_vec type=intra direction=WAR true

                    ap_uint<72> op1_mtx[16];
                    ap_uint<72> op2_mtx[16];
                    #pragma HLS array_partition variable=op1_mtx complete
                    #pragma HLS array_partition variable=op2_mtx complete

                    ap_uint<1024> recv_pkt;

                    if(stage == 3) {
                        recv_pkt = fifo_context.read();
                    }

                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS unroll
                        if(stage == 3){
                            op1_mtx[ii] = ap_uint<72>(ap_uint<36>((ap_int<4>(2), W[k*8+ii%8][j*2+ii/8])));
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), recv_pkt(ii*64+63, ii*64)));
                        } else if(stage != 2) {
                            op1_mtx[ii] = ap_uint<72>(ap_uint<36>((ap_int<4>(2), W[j*16+ii][k])));
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), X[i*16+ii][k]));
                        } else {
                            op1_mtx[ii] = ap_uint<72>(ap_uint<36>((ap_int<4>(2), cache_attn[k][ii])));
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), scratchpad[k*8+ii/2][j*2+(ii%2)])); 
                        }
                    }

                    if(stage < 2){
                        ap_uint<1024> send_pkt = ap_uint<1024>((
                            op2_mtx[0](63,0), op2_mtx[1](63,0), op2_mtx[2](63,0), op2_mtx[3](63,0), op2_mtx[4](63,0), op2_mtx[5](63,0), op2_mtx[6](63,0), op2_mtx[7](63,0),
                            op2_mtx[8](63,0), op2_mtx[9](63,0), op2_mtx[10](63,0), op2_mtx[11](63,0), op2_mtx[12](63,0), op2_mtx[13](63,0), op2_mtx[14](63,0), op2_mtx[15](63,0)
                        ));
                        fifo_X_out.write(send_pkt);
                    }

                    for(int kk = 0; kk < 16; kk++){
                        #pragma HLS unroll
                        for(int l = 0; l < 16; l++){
                            #pragma HLS unroll
                            ap_int<22> res0 = acc_vec[k%2][0][kk][l];
                            ap_int<22> res1 = acc_vec[k%2][1][kk][l];
                            ap_int<22> res2 = acc_vec[k%2][2][kk][l];
                            ap_int<8> a0; ap_int<8> a1; ap_int<8> a2;
                            ap_int<8> b0; ap_int<8> b1; ap_int<8> b2;
                            ap_int<8> a0_1; ap_int<8> a1_1; ap_int<8> a2_1;
                            ap_int<8> b0_1; ap_int<8> b1_1; ap_int<8> b2_1;
                            ap_int<8> a0_2; ap_int<8> a1_2; ap_int<8> a2_2;
                            ap_int<8> b0_2; ap_int<8> b1_2; ap_int<8> b2_2;
                            (b2_2, b1_2, b0_2, b2_1, b1_1, b0_1, b2, b1, b0) = op2_mtx[kk];
                            a0 = ap_int<4>(op1_mtx[l](3, 0));
                            a1 = ap_int<4>(op1_mtx[l](7, 4));
                            a2 = ap_int<4>(op1_mtx[l](11, 8));
                            a0_1 = ap_int<4>(op1_mtx[l](15, 12));
                            a1_1 = ap_int<4>(op1_mtx[l](19, 16));
                            a2_1 = ap_int<4>(op1_mtx[l](23, 20));
                            a0_2 = ap_int<4>(op1_mtx[l](27, 24));
                            a1_2 = ap_int<4>(op1_mtx[l](31, 28));
                            a2_2 = ap_int<4>(op1_mtx[l](35, 32));
                            res0 = (((a0 * b0) + (a1 * b1)) + (a2 * b2)) + res0;
                            res1 = (((a0_1 * b0_1) + (a1_1 * b1_1)) + (a2_1 * b2_1)) + res1;
                            res2 = (((a0_2 * b0_2) + (a1_2 * b1_2)) + (a2_2 * b2_2)) + res2;
                            acc_vec[k%2][0][kk][l] = res0;
                            acc_vec[k%2][1][kk][l] = res1;
                            acc_vec[k%2][2][kk][l] = res2;
                        }
                    }
                }

                ap_int<24> acc_final[16][16];
                #pragma HLS array_partition variable=acc_final dim=1 complete
                #pragma HLS array_partition variable=acc_final dim=2 complete

                for(int ii = 0; ii < 16; ii++){
                    #pragma HLS unroll
                    for(int k = 0; k < 16; k++){
                        #pragma HLS unroll
                        acc_final[ii][k] = -(k_bound << 2);
                    }
                }

        reduction:
                for(int k = 0; k < 2; k++){
                    for(int ii = 0; ii < 3; ii++){
                        for(int kk = 0; kk < 16; kk++){
                            #pragma HLS unroll
                            for(int l = 0; l < 16; l++){
                                #pragma HLS unroll
                                acc_final[kk][l] += acc_vec[k][ii][kk][l];
                            }
                        }
                    }
                }

                if(stage == 0){
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS unroll
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            int offset = k%8;
                            scratchpad[i*16+ii][j*2+k/8](offset*8+7, offset*8) = ap_int<8>(acc_final[k][ii] >> 5); 
                        }
                    }
                } else if (stage == 2){
                    for(int ii = 0; ii < 2; ii++){
                        #pragma HLS pipeline II=1 style=stp
                        ap_uint<1024> tmp;
                        for(int jj = 0; jj < 8; jj++){
                            #pragma HLS unroll
                            for(int k = 0; k < 16; k++){
                                #pragma HLS unroll
                                tmp((jj*16+k)*8+7, (jj*16+k)*8) = ap_int<8>(acc_final[k][ii*8+jj] >> 5);
                            }
                        }
                        fifo_O_out.write(tmp);
                    }
                } else if (stage == 1) {
                    for(int ii = 0; ii < 16; ii++){
                        ap_uint<128> tmp;
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            tmp(k*8+7, k*8) = ap_int<8>(acc_final[ii][k] >> 5);
                        }
                        fifo_to_acc0.write(tmp);
                    }
                } else {
                    final_acc:
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS pipeline II=1 style=stp
                        #pragma HLS dependence variable=X type=inter false
                        ap_uint<512> tmp_recv = fifo_reduce_recv.read();
                        ap_uint<512> tmp_send;
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            ap_int<32> tmp = acc_final[ii][k] + ap_int<22>(tmp_recv(k*32+21, k*32));
                            tmp += ap_int<8>(X[i*16+ii][j*2+k/8]((k%8)*8+7, (k%8)*8));
                            tmp_send(k*32+31, k*32) = tmp;
                        }
                        fifo_res_send.write(tmp_send);
                    }
                }                
            }
        }
    }
    // fifo_fin.write(true);

    // write out for debug
// write:
//     for(int i = 0; i < L; i++){
//         for(int j = 0; j < D_div_8; j++){
//             #pragma HLS pipeline II=1 style=stp
//             fifo_write.write(X[i][j]);
//         }
//     }
}

void residual(
    const int L,
    tapa::istream<ap_uint<512>>& fifo_res_in,
    tapa::ostream<ap_uint<512>>& fifo_res_out
){
    ap_uint<32> res_buffer[16][16];
    #pragma HLS array_partition variable=res_buffer complete dim=1
    #pragma HLS array_partition variable=res_buffer complete dim=2

    for(int i = 0; i < (L >> 5); i++){
        for(int j = 0; j < D_div_16; j++){
            read:
            for(int k = 0; k < 16;){
                #pragma HLS pipeline II=1 style=stp
                ap_uint<512> tmp;
                bool success = fifo_res_in.try_read(tmp);
                if(success){
                    for(int l = 0; l < 16; l++){
                        #pragma HLS unroll
                        res_buffer[k][l] = ap_uint<32>(tmp(l*32+31, l*32));
                    }
                    k++;
                }
            }
            transpose:
            for(int k = 0; k < 16; k++){
                #pragma HLS pipeline II=1 style=stp
                ap_uint<512> tmp;
                for(int l = 0; l < 16; l++){
                    #pragma HLS unroll
                    tmp(l*32+31, l*32) = ap_uint<32>(res_buffer[l][k]); 
                }
                fifo_res_out.write(tmp);
            }
        }
    }
}


void temporal_acc1(
    const int L,
    tapa::istream<int>& fifo_len_in,
    tapa::ostream<int>& fifo_len_out,
    tapa::istream<ap_uint<1024>>& fifo_X_in,
    tapa::ostream<ap_uint<1024>>& fifo_X_out, // 8-bit activation
    tapa::istream<ap_uint<512>>& fifo_W_in,
    tapa::ostream<ap_uint<512>>& fifo_W_out, // 4-bit weight
    tapa::ostream<ap_uint<128>>& fifo_to_acc0,
    tapa::istream<ap_uint<128>>& fifo_from_sfu,
    tapa::ostream<ap_uint<1024>>& fifo_O_out,
    tapa::istream<ap_uint<1024>>& fifo_context,
    tapa::istream<ap_uint<512>>& fifo_reduce_recv,
    tapa::ostream<ap_uint<512>>& fifo_reduce_send
){

    ap_uint<64> scratchpad[MAX_SEQ_LEN][D_head_div_8]; // 8 bit
    #pragma HLS array_partition variable=scratchpad cyclic dim=1 factor=16
    #pragma HLS array_partition variable=scratchpad cyclic dim=2 factor=2
    #pragma HLS bind_storage variable=scratchpad type=ram_2p impl=uram

    // ap_uint<64> scratchpad_out[MAX_SEQ_LEN][D_head_div_8];
    // #pragma HLS array_partition variable=scratchpad_out cyclic dim=1 factor=16
    // #pragma HLS array_partition variable=scratchpad_out cyclic dim=2 factor=2

    for(int stage_i = 0; stage_i < 13; stage_i++){

        // stage 0: WvX
        // stage 1: WkX1 -> acc0
        // stage 2: Softmax(QK)V <- acc0
        // stage 3: WoO

        ap_uint<32> W[D_head_mul_4][D_div_8]; // 4 bit
        #pragma HLS array_partition variable=W cyclic dim=1 factor=16

        const int start = fifo_len_in.read();
        const int end = fifo_len_in.read();
        fifo_len_out.write(start);
        fifo_len_out.write(end);

        const int stage = (stage_i < 12) ? (stage_i % 3) : 3;

        // load weights and forward
        if(stage != 2) {
            const int bound = (stage == 3) ? D_head : D_head_div_4;
            for(int i = 0; i < bound; i++){
                load_weight:
                for(int j = 0; j < D_div_8;){
                    if(!fifo_W_in.empty()){
                        ap_uint<512> val; fifo_W_in.try_read(val);

                        for(int k = 0; k < 4; k++){
                            #pragma HLS unroll
                            W[i*4+k][j] = ap_uint<32>(val(k*32+31, k*32));
                        }
                        val = ap_uint<512>(val >> 128);
                        fifo_W_out.write(val);
                        j++;
                    }
                }
            }
        }

        int k_bound = (stage == 2) ? (L >> 3) : D_div_8;
        k_bound = (stage == 3) ? D_head_div_2 : k_bound;
        int j_bound = (stage == 3) ? D_div_16 : D_head_div_16;
        
        for(int i = (start >> 4); i < (end >> 4); i++){ // make sure L is multiple of 4

            ap_uint<32> cache_attn[MAX_SEQ_LEN_div_8][16];
            #pragma HLS array_partition variable=cache_attn dim=2 complete
            #pragma HLS array_partition variable=cache_attn dim=1 cyclic factor=2

            if(stage == 2){
                for(int ii = 0; ii < (L >> 3); ii++){
                    ap_uint<32> fuse_reg[16];
                    load_attn:
                    for(int offset = 0; offset < 8;){
                        #pragma HLS pipeline II=1 style=stp
                        if(!fifo_from_sfu.empty()){
                            ap_uint<128> val; fifo_from_sfu.try_read(val);
                            for(int k = 0; k < 16; k++){
                                #pragma HLS unroll
                                fuse_reg[k](offset*4+3, offset*4) = ap_int<4>(val(k*8+3, k*8));
                            }
                            offset++;
                        }
                    }
                    for(int k = 0; k < 16; k++){
                        #pragma HLS unroll
                        cache_attn[ii][k] = fuse_reg[k];
                    }
                }
            }

            for(int j = 0; j < j_bound; j++){
                #pragma HLS loop_flatten off

                ap_int<22> acc_vec[2][3][16][16];
                #pragma HLS array_partition variable=acc_vec dim=1 complete
                #pragma HLS array_partition variable=acc_vec dim=2 complete
                #pragma HLS array_partition variable=acc_vec dim=3 complete
                #pragma HLS array_partition variable=acc_vec dim=4 complete

                for(int l = 0; l < 2; l++){
                    #pragma HLS unroll
                    for(int ii = 0; ii < 3; ii++){
                        #pragma HLS unroll
                        for(int kk = 0; kk < 16; kk++){
                            #pragma HLS unroll
                            for(int k = 0; k < 16; k++){
                                #pragma HLS unroll
                                acc_vec[l][ii][kk][k] = 0;
                            }
                        }
                    }
                }
        compute:
                for(int k = 0; k < k_bound; k++){
                    #pragma HLS pipeline II=1 style=stp
                    #pragma HLS dependence variable=acc_vec type=intra direction=RAW false
                    #pragma HLS dependence variable=acc_vec type=intra direction=WAR true

                    ap_uint<72> op1_mtx[16];
                    ap_uint<72> op2_mtx[16];
                    #pragma HLS array_partition variable=op1_mtx complete
                    #pragma HLS array_partition variable=op2_mtx complete

                    ap_uint<1024> recv_pkt;

                    if(stage == 3) {
                        recv_pkt = fifo_context.read();
                    } else if(stage != 2) {
                        recv_pkt = fifo_X_in.read();
                        fifo_X_out.write(recv_pkt);
                    }

                    for(int ii = 0; ii < 16; ii++){ //TODO: change logic
                        #pragma HLS unroll
                        if(stage == 3){
                            op1_mtx[ii] = ap_uint<72>(ap_uint<36>((ap_int<4>(2), W[k*8+ii%8][j*2+ii/8])));
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), recv_pkt(ii*64+63, ii*64)));
                        } else if(stage != 2) {
                            op1_mtx[ii] = ap_uint<72>(ap_uint<36>((ap_int<4>(2), W[j*16+ii][k])));
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), recv_pkt(ii*64+63, ii*64)));
                        } else {
                            op1_mtx[ii] = ap_uint<72>(ap_uint<36>((ap_int<4>(2), cache_attn[k][ii])));
                            op2_mtx[ii] = ap_uint<72>((ap_int<8>(2), scratchpad[k*8+ii/2][j*2+(ii%2)])); 
                        }
                    }
                    
                    for(int kk = 0; kk < 16; kk++){
                        #pragma HLS unroll
                        for(int l = 0; l < 16; l++){
                            #pragma HLS unroll
                            ap_int<22> res0 = acc_vec[k%2][0][kk][l];
                            ap_int<22> res1 = acc_vec[k%2][1][kk][l];
                            ap_int<22> res2 = acc_vec[k%2][2][kk][l];
                            ap_int<8> a0; ap_int<8> a1; ap_int<8> a2;
                            ap_int<8> b0; ap_int<8> b1; ap_int<8> b2;
                            ap_int<8> a0_1; ap_int<8> a1_1; ap_int<8> a2_1;
                            ap_int<8> b0_1; ap_int<8> b1_1; ap_int<8> b2_1;
                            ap_int<8> a0_2; ap_int<8> a1_2; ap_int<8> a2_2;
                            ap_int<8> b0_2; ap_int<8> b1_2; ap_int<8> b2_2;
                            (b2_2, b1_2, b0_2, b2_1, b1_1, b0_1, b2, b1, b0) = op2_mtx[kk];
                            a0 = ap_int<4>(op1_mtx[l](3, 0));
                            a1 = ap_int<4>(op1_mtx[l](7, 4));
                            a2 = ap_int<4>(op1_mtx[l](11, 8));
                            a0_1 = ap_int<4>(op1_mtx[l](15, 12));
                            a1_1 = ap_int<4>(op1_mtx[l](19, 16));
                            a2_1 = ap_int<4>(op1_mtx[l](23, 20));
                            a0_2 = ap_int<4>(op1_mtx[l](27, 24));
                            a1_2 = ap_int<4>(op1_mtx[l](31, 28));
                            a2_2 = ap_int<4>(op1_mtx[l](35, 32));
                            res0 = (((a0 * b0) + (a1 * b1)) + (a2 * b2)) + res0;
                            res1 = (((a0_1 * b0_1) + (a1_1 * b1_1)) + (a2_1 * b2_1)) + res1;
                            res2 = (((a0_2 * b0_2) + (a1_2 * b1_2)) + (a2_2 * b2_2)) + res2;
                            acc_vec[k%2][0][kk][l] = res0;
                            acc_vec[k%2][1][kk][l] = res1;
                            acc_vec[k%2][2][kk][l] = res2;
                        }
                    }
                }

                ap_int<24> acc_final[16][16];
                #pragma HLS array_partition variable=acc_final dim=1 complete
                #pragma HLS array_partition variable=acc_final dim=2 complete

                for(int ii = 0; ii < 16; ii++){
                    #pragma HLS unroll
                    for(int k = 0; k < 16; k++){
                        #pragma HLS unroll
                        acc_final[ii][k] = 0;
                    }
                }

        reduction:
                for(int k = 0; k < 2; k++){
                    for(int ii = 0; ii < 3; ii++){
                        for(int kk = 0; kk < 16; kk++){
                            #pragma HLS unroll
                            for(int l = 0; l < 16; l++){
                                #pragma HLS unroll
                                acc_final[kk][l] += acc_vec[k][ii][kk][l];
                            }
                        }
                    }
                }

                if(stage == 0){
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS unroll
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            int offset = k%8;
                            scratchpad[i*16+ii][j*2+k/8](offset*8+7, offset*8) = ap_int<8>(acc_final[k][ii] >> 5); 
                        }
                    }
                } else if (stage == 2){
                    for(int ii = 0; ii < 2; ii++){
                        #pragma HLS pipeline II=1 style=stp
                        ap_uint<1024> tmp;
                        for(int jj = 0; jj < 8; jj++){
                            #pragma HLS unroll
                            for(int k = 0; k < 16; k++){
                                #pragma HLS unroll
                                tmp((jj*16+k)*8+7, (jj*16+k)*8) = ap_int<8>(acc_final[k][ii*8+jj] >> 5);
                            }
                        }
                        fifo_O_out.write(tmp);
                    }
                } else if (stage == 1){
                    for(int ii = 0; ii < 16; ii++){
                        ap_uint<128> tmp;
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            tmp(k*8+7, k*8) = ap_int<8>(acc_final[ii][k] >> 5);
                        }
                        fifo_to_acc0.write(tmp);
                    }
                } else {
                    final_acc:
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS pipeline II=1 style=stp
                        ap_uint<512> tmp_recv = fifo_reduce_recv.read();
                        ap_uint<512> tmp;
                        for(int k = 0; k < 16; k++){
                            #pragma HLS unroll
                            acc_final[ii][k] += ap_int<22>(tmp_recv(k*32+21, k*32));
                            tmp(k*32+21, k*32) = acc_final[ii][k];
                        }
                        fifo_reduce_send.write(tmp);
                    }
                }         
            }
        }
    }

    // write out for debug
// write:
//     for(int i = 0; i < L; i++){
//         for(int j = 0; j < D_head_div_8; j++){
//             #pragma HLS pipeline II=1 style=stp
//             fifo_O_out.write(scratchpad_out[i][j]);
//         }
//     }
}

void sfu_buffer( // double buffering
    const int L,
    tapa::istream<ap_uint<512>>& fifo_data_in,
    tapa::ostream<ap_uint<512>>& fifo_data_out
){
    for(int stage = 0; stage < 4; stage++){

        for(int l = 0; l < (L >> 5); l++){
            float sum[8][16];
            float cache[MAX_SEQ_LEN][16];
            #pragma HLS array_partition variable=cache dim=2 complete
            #pragma HLS array_partition variable=sum dim=2 complete
            
            for(int i = 0; i < 8; i++){
                for(int j = 0; j < 16; j++){
                    #pragma HLS unroll
                    sum[i][j] = 0.0;
                }
            }

        acc:
            for(int i = 0; i < L; i++){
                #pragma HLS pipeline II=1 style=stp
                #pragma HLS dependence false variable=sum
                #pragma HLS dependence true variable=sum distance=8
                ap_uint<512> tmp = fifo_data_in.read();
                for(int k = 0; k < 16; k++){
                    #pragma HLS unroll
                    float res = tapa::bit_cast<float>(ap_int<32>(tmp(k*32+31, k*32)));
                    sum[i%8][k] += res;
                    cache[i][k] = res;
                }
            }

        reduce:
            for(int i = 1; i < 8; i++){
                for(int j = 0; j < 8; j++){
                    #pragma HLS pipeline II=1 style=stp
                    #pragma HLS dependence true variable=sum distance=8
                    for(int k = 0; k < 2; k++){
                        sum[0][j*2+k] += sum[i][j*2+k];
                    }
                }
            }

            ap_uint<512> tmp;
            for(int i = 0; i < 16; i++){
                #pragma HLS unroll
                tmp(i*32+31, i*32) = tapa::bit_cast<ap_uint<32>>(sum[0][i]);
            }
            fifo_data_out.write(tmp);

        write:
            for(int i = 0; i < L; i++){
                #pragma HLS pipeline II=1 style=stp
                ap_uint<512> tmp;
                for(int j = 0; j < 16; j++){
                    #pragma HLS unroll
                    tmp(j*32+31, j*32) = tapa::bit_cast<ap_uint<32>>(cache[i][j]);
                }
                fifo_data_out.write(tmp);
            }

        }
    }

}

void sfu_buffer_slr0( // double buffering
    const int L,
    tapa::istream<ap_uint<512>>& fifo_data_in_exp,
    tapa::istream<ap_uint<512>>& fifo_data_in_ln, 
    tapa::ostream<ap_uint<512>>& fifo_data_out
){
    for(int stage = 0; stage < 5; stage++){

        const int hidden_bound = (stage < 4) ? L : D;

        for(int l = 0; l < (L >> 5); l++){
            float sum[8][16];
            float var[8][16];
            float cache[MAX_SEQ_LEN][16];
            #pragma HLS array_partition variable=cache dim=2 complete
            #pragma HLS array_partition variable=sum dim=2 complete
            #pragma HLS array_partition variable=var dim=2 complete
            
            for(int i = 0; i < 8; i++){
                for(int j = 0; j < 16; j++){
                    #pragma HLS unroll
                    sum[i][j] = 0.0;
                    var[i][j] = 0.0;
                }
            }

        acc:
            for(int i = 0; i < hidden_bound; i++){
                #pragma HLS pipeline II=1 style=stp
                #pragma HLS dependence false variable=sum
                #pragma HLS dependence true variable=sum distance=8
                ap_uint<512> tmp;
                if(stage < 4) {
                    tmp = fifo_data_in_exp.read();
                } else {
                    tmp = fifo_data_in_ln.read();
                }
                
                for(int k = 0; k < 16; k++){
                    #pragma HLS unroll
                    float res = tapa::bit_cast<float>(ap_int<32>(tmp(k*32+31, k*32)));
                    sum[i%8][k] += res;
                    if(stage == 4) var[i%8][k] += (res * res);
                    cache[i][k] = res;
                }
            }

        reduce:
            for(int i = 1; i < 8; i++){
                for(int j = 0; j < 8; j++){
                    #pragma HLS pipeline II=1 style=stp
                    #pragma HLS dependence true variable=sum distance=8
                    #pragma HLS dependence true variable=var distance=8
                    for(int k = 0; k < 2; k++){
                        sum[0][j*2+k] += sum[i][j*2+k];
                        if(stage == 4) var[0][j*2+k] += var[i][j*2+k];
                    }
                }
            }

            ap_uint<512> tmp;
            ap_uint<512> tmp_var;
            for(int i = 0; i < 16; i++){
                #pragma HLS unroll
                tmp(i*32+31, i*32) = tapa::bit_cast<ap_uint<32>>(sum[0][i]);
                if(stage == 4) tmp_var(i*32+31, i*32) = tapa::bit_cast<ap_uint<32>>(var[0][i]);
            }
            fifo_data_out.write(tmp);
            if(stage == 4) fifo_data_out.write(tmp_var);

        write:
            for(int i = 0; i < hidden_bound; i++){
                #pragma HLS pipeline II=1 style=stp
                ap_uint<512> tmp;
                for(int j = 0; j < 16; j++){
                    #pragma HLS unroll
                    tmp(j*32+31, j*32) = tapa::bit_cast<ap_uint<32>>(cache[i][j]);
                }
                fifo_data_out.write(tmp);
            }

        }
    }

}

void sfu_acc_exp(
    const int L,
    tapa::istream<ap_uint<512>>& fifo_data_in,
    tapa::ostreams<ap_uint<512>, 2>& fifo_buf
) {
    for(int stage = 0; stage < 4; stage++){

        for(int l = 0; l < (L >> 4); l++){
            exp_acc:
            for(int i = 0; i < L;){
                #pragma HLS pipeline II=1 style=stp
                if(!fifo_data_in.empty()){
                    ap_uint<512> tmp; fifo_data_in.try_read(tmp);
                    ap_uint<512> tmp_o;
                    for(int k = 0; k < 16; k++){
                        #pragma HLS unroll
                        int res = tapa::bit_cast<int>(ap_int<32>(tmp(k*32+31, k*32)));
                        float res_exp = 0.0;
                        res_exp = hls::exp(ap_int<32>(res >> 10));
                        tmp_o(k*32+31, k*32) = tapa::bit_cast<ap_uint<32>>(res_exp);
                    }
                    fifo_buf[l%2].write(tmp_o);
                    i++;
                }
            }
        }
    }
}

void sfu_gelu(
    const int L,
    tapa::istream<ap_uint<512>>& fifo_ffn,
    tapa::ostream<ap_uint<128>>& fifo_out
){
    for(int i = 0; i < (L >> 4); i++){
        for(int j = 0; j < D_ffn;){
            if(!fifo_ffn.empty()){
                ap_uint<512> tmp; fifo_ffn.try_read(tmp);
                ap_uint<128> tmp_out;
                for(int k = 0; k < 16; k++){
                    // piecewise linear approximation: https://github.com/cornell-zhang/allo-pldi24-artifact/blob/main/llm/gpt_region_3.cpp#L335
                    float val = (float) tapa::bit_cast<int>(ap_int<32>(tmp(k*32+31, k*32)));
                    float outp_data = 0.0;
                    if (val < -3)
                        outp_data = 0;
                    else if(val < -1)
                        outp_data = -0.0773 * (val + 3) - 0.004;
                    else if(val < 0)
                        outp_data = 0.1587 * val;
                    else if(val < 1)
                        outp_data = 0.8413 * val;
                    else if(val < 3)
                        outp_data = 1.0773 * (val - 1) + 0.8413;
                    else
                        outp_data = val;
                    tmp_out(k*8+7, k*8) = ap_int<8>((int) (outp_data) >> 8);
                }
                fifo_out.write(tmp_out);
                j++;
            }
        }
    }
}

void sfu_norm(
    const int L,
    tapa::istreams<ap_uint<512>, 2>& fifo_buf,
    tapa::ostream<ap_uint<128>>& fifo_data_out
){
    for(int stage = 0; stage < 4; stage++){

        for(int l = 0; l < (L >> 4); l++){
            float sum[16];
            #pragma HLS array_partition variable=sum complete

            ap_uint<512> tmp_in = fifo_buf[l%2].read();

            for(int i = 0; i < 16; i++){
                #pragma HLS unroll
                sum[i] = 32.0 / tapa::bit_cast<float>(ap_uint<32>(tmp_in(i*32+31, i*32)));
            }

            for(int i = 0; i < L;){
                #pragma HLS pipeline II=1 style=stp
                if(!fifo_buf[l%2].empty()){
                    ap_uint<512> tmp_cache; fifo_buf[l%2].try_read(tmp_cache);
                    ap_uint<128> tmp;
                    for(int j = 0; j < 16; j++){
                        #pragma HLS unroll
                        ap_int<8> res = (int) (tapa::bit_cast<float>(ap_uint<32>(tmp_cache(j*32+31, j*32))) * sum[j]);
                        tmp(j*8 + 7, j*8) = res;
                    }
                    fifo_data_out.write(tmp);
                    i++;
                }
            }
        }
    }
}

void sfu_norm_slr0(
    const int L,
    tapa::istreams<ap_uint<512>, 2>& fifo_buf,
    tapa::ostream<ap_uint<128>>& fifo_data_out,
    tapa::ostream<ap_uint<128>>& fifo_data_off
){
    for(int stage = 0; stage < 5; stage++){
        const int hidden_bound = (stage < 4) ? L : D;

        for(int l = 0; l < (L >> 4); l++){
            float sum[16];
            float mean[16];
            float var[16];
            #pragma HLS array_partition variable=sum complete
            #pragma HLS array_partition variable=mean complete
            #pragma HLS array_partition variable=var complete

            const int fifo_idx = l%2;

            ap_uint<512> tmp_in = fifo_buf[fifo_idx].read();
            ap_uint<512> tmp_var;
            if(stage == 4) tmp_var = fifo_buf[fifo_idx].read();

            if(stage == 4){
                for(int i = 0; i < 16; i++){
                    #pragma HLS unroll
                    mean[i] = tapa::bit_cast<float>(ap_uint<32>(tmp_in(i*32+31, i*32))) / D;
                    var[i] = 1024 / (tapa::bit_cast<float>(ap_uint<32>(tmp_var(i*32+31, i*32))) / D - mean[i]*mean[i] + (float) 0.00001);
                }
            } else {
                for(int i = 0; i < 16; i++){
                    #pragma HLS unroll
                    sum[i] = 32.0 / tapa::bit_cast<float>(ap_uint<32>(tmp_in(i*32+31, i*32)));
                }
            }

            for(int i = 0; i < hidden_bound;){
                #pragma HLS pipeline II=1 style=stp
                if(!fifo_buf[fifo_idx].empty()){
                    ap_uint<512> tmp_cache; fifo_buf[fifo_idx].try_read(tmp_cache);
                    ap_uint<128> tmp;
                    for(int j = 0; j < 16; j++){
                        #pragma HLS unroll
                        ap_int<8> res;
                        float op1; float op2;
                        if(stage == 4){
                            op1 = tapa::bit_cast<float>(ap_uint<32>(tmp_cache(j*32+31, j*32))) - mean[j];
                            op2 = std::sqrt(var[j]);
                        } else {
                            op1 = tapa::bit_cast<float>(ap_uint<32>(tmp_cache(j*32+31, j*32)));
                            op2 = sum[j];
                        }
                        res = (int) (op1 * op2);
                        tmp(j*8 + 7, j*8) = res;
                    }
                    if(stage == 4) {
                        fifo_data_off.write(tmp);
                    } else {
                        fifo_data_out.write(tmp);
                    }
                    i++;
                }
            }
        }
    }
}

void context_buffer(
    const int L,
    tapa::istream<ap_uint<1024>>& fifo_context,
    tapa::ostream<ap_uint<1024>>& fifo_to_acc0,
    tapa::ostream<ap_uint<1024>>& fifo_to_acc1
){
    ap_uint<64> context[MAX_SEQ_LEN][D_head_div_2];
    #pragma HLS array_partition variable=context cyclic dim=1 factor=32
    #pragma HLS bind_storage variable=context type=ram_2p impl=uram

    for(int stage = 0; stage < 4; stage++){
        for(int i = 0; i < (L >> 4); i++){
            for(int j = stage * D_head_div_8; j < (stage + 1) * D_head_div_8;){
                if(!fifo_context.empty()){
                    ap_uint<1024> tmp; fifo_context.try_read(tmp);
                    for(int ii = 0; ii < 16; ii++){
                        #pragma HLS unroll
                        context[i*16+ii][j] = tmp(ii*64+63, ii*64);
                    }
                    j++;
                }
            }
        }
    }

    // NOTE: change it to write to HBM for debugging
    // write ops to acc0 and acc1 in parallel
    for(int i = 0; i < (L >> 5); i++){
        for(int l = 0; l < D_div_16; l++){
            for(int j = 0; j < D_head_div_2; j++){
                ap_uint<1024> tmp_acc0;
                ap_uint<1024> tmp_acc1;
                for(int k = 0; k < 16; k++){
                    #pragma HLS unroll
                    tmp_acc0(k*64+63, k*64) = context[i*32+k][j];
                    tmp_acc1(k*64+63, k*64) = context[i*32+16+k][j];
                }
                fifo_to_acc0.write(tmp_acc0);
                fifo_to_acc1.write(tmp_acc1);
            }
        }
    }
}

void ffn_buffer(
    const int L,
    tapa::istream<ap_uint<128>>& fifo_ffn_in,
    tapa::ostream<ap_uint<1024>>& fifo_ffn_out
){
    ap_uint<64> X[MAX_SEQ_LEN][D_div_8]; // 8 bit
    #pragma HLS array_partition variable=X cyclic dim=1 factor=16
    #pragma HLS bind_storage variable=X type=ram_2p impl=uram

    for(int i = 0; i < (L >> 4); i++){
        for(int j = 0; j < D_div_8; j++){
            ap_uint<64> fuse_reg[16];
            #pragma HLS array_partition variable=fuse_reg complete

            for(int l = 0; l < 8;){
                #pragma HLS pipeline II=1

                if(!fifo_ffn_in.empty()){
                    ap_uint<128> tmp; fifo_ffn_in.try_read(tmp);
                    for(int k = 0; k < 16; k++){
                        #pragma HLS unroll
                        fuse_reg[k](l*8+7, l*8) = tmp(k*8+7, k*8);
                    }
                    l++;
                }
            }
            for(int k = 0; k < 16; k++){
                #pragma HLS unroll
                X[i*16+k][j] = fuse_reg[k];
            }
        }
    }

    for(int i = 0; i < (L >> 4); i++){
        for(int iter = 0; iter < D_div_16; iter++){
            for(int j = 0; j < D_div_8; j++){
                ap_uint<1024> tmp;
                for(int k = 0; k < 16; k++){
                    #pragma HLS unroll
                    tmp(k*64+63, k*64) = X[i*16+k][j];
                }
                fifo_ffn_out.write(tmp);
            }
        }
    }
}


// TODO: remove after implementation. debug only
void aggregator(
    const int L,
    tapa::istream<ap_uint<512>>& fifo_in,
    tapa::istream<ap_uint<512>>& fifo_data,
    tapa::ostream<ap_uint<512>>& fifo_out
){
    for(int i = 0; i < (L >> 4); i++){
        for(int j = 0; j < D; j++){
            auto op1 = fifo_in.read();
            auto op2 = fifo_data.read();
            ap_uint<512> out = op1 ^ op2;
            fifo_out.write(out);
        }
    } 
}

void write_trash(
    const int L,
    tapa::ostream<ap_uint<512>>& fifo_out
){
    for(int i = 0; i < (L >> 4); i++){
        for(int j = 0; j < D; j++){
            fifo_out.write(0);
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

void opt_kernel(
    const int L,
    const int L_out,
    const int seq_len,
    // tapa::mmap<int> inst, // inst[0] = L, inst[1] = reload_weight
    tapa::mmap<ap_uint<512>> X_acc0,
    tapa::mmap<ap_uint<512>> X_acc1,
    tapa::mmap<ap_uint<512>> W_acc0,
    tapa::mmap<ap_uint<512>> W_acc1,
    tapa::mmap<ap_uint<512>> acc0_out,
    // tapa::mmap<ap_uint<64>> acc1_out,
    tapa::mmap<int> cycle_count
){
    tapa::streams<int, NUM_SLR+1, 4> fifo_inst_acc0("fifo_inst_acc0");
    tapa::streams<int, NUM_SLR+1, 4> fifo_inst_acc1("fifo_inst_acc1");
    tapa::stream<ap_uint<512>, 16> fifo_X_acc0_slr0("fifo_X_acc0_slr0");
    tapa::stream<ap_uint<512>, 16> fifo_X_acc1_slr0("fifo_X_acc1_slr0");
    tapa::streams<ap_uint<1024>, NUM_SLR, 4> fifo_X_acc0("fifo_X_acc0");
    tapa::streams<ap_uint<1024>, NUM_SLR, 4> fifo_X_acc1("fifo_X_acc1");
    tapa::streams<ap_uint<512>, NUM_SLR+1, 8> fifo_W_acc0("fifo_W_acc0");
    tapa::streams<ap_uint<512>, NUM_SLR+1, 8> fifo_W_acc1("fifo_W_acc1");
    // tapa::streams<ap_uint<512>, NUM_SLR, 4> fifo_acc0_out("fifo_acc0_out");
    tapa::streams<ap_uint<512>, NUM_SLR> fifo_acc0_to_sfu("fifo_acc0_to_sfu");
    tapa::streams<ap_uint<512>, NUM_SLR*2> fifo_sfu_buf_in("fifo_sfu_buf_in");
    tapa::streams<ap_uint<512>, NUM_SLR*2> fifo_sfu_buf_out("fifo_sfu_buf_out");
    // tapa::streams<ap_uint<64>, NUM_SLR> fifo_acc1_out("fifo_acc1_out");
    tapa::streams<ap_uint<128>, NUM_SLR, 2> fifo_from_acc1_to_acc0("fifo_from_acc1_to_acc0");
    tapa::streams<ap_uint<128>, NUM_SLR, 2> fifo_from_sfu_to_acc1("fifo_from_sfu_to_acc1");
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::streams<ap_uint<1024>, NUM_SLR> fifo_context("fifo_context");
    tapa::streams<ap_uint<1024>, NUM_SLR> fifo_cont_to_acc0("fifo_cont_to_acc0");
    tapa::streams<ap_uint<1024>, NUM_SLR> fifo_cont_to_acc1("fifo_cont_to_acc1");
    tapa::streams<ap_uint<512>, NUM_SLR> fifo_reduce_acc0("fifo_reduce_acc0");
    tapa::streams<ap_uint<512>, NUM_SLR> fifo_reduce_acc1("fifo_reduce_acc1");

    // tapa::stream<ap_uint<128>> fifo_acc0_out("fifo_acc0_out");
    // tapa::stream<ap_uint<64>> fifo_acc1_out("fifo_acc1_out");

    tapa::stream<ap_uint<512>> fifo_res_acc0("fifo_res_acc0");
    tapa::stream<ap_uint<512>> fifo_res_acc1("fifo_res_acc1");
    tapa::stream<ap_uint<512>, D> fifo_ln_acc0("fifo_ln_acc0");
    tapa::stream<ap_uint<512>, D> fifo_ln_acc1("fifo_ln_acc1");

    tapa::stream<ap_uint<128>> fifo_ffn_buffer_in("fifo_ffn_buffer_in");
    tapa::stream<ap_uint<1024>> fifo_ffn_buffer_out("fifo_ffn_buffer_out");

    tapa::streams<ap_uint<512>, NUM_SLR> fifo_gelu_in("fifo_gelu_in");

    tapa::streams<ap_uint<512>, NUM_SLR+1> fifo_debug("fifo_debug");

    tapa::task()
        .invoke<tapa::join>(read_inst, seq_len, fifo_inst_acc0, fifo_inst_acc1)
        .invoke<tapa::join>(read_W, WEIGHT_SIZE, W_acc0, fifo_W_acc0)
        .invoke<tapa::join>(read_W, TOTAL_WEIGHT_SIZE, W_acc1, fifo_W_acc1)
        .invoke<tapa::join>(read_X, L, X_acc0, fifo_X_acc0_slr0)
        .invoke<tapa::join>(read_X, L, X_acc1, fifo_X_acc1_slr0)
        .invoke<tapa::join>(
            temporal_acc0_slr0,
            seq_len,
            fifo_inst_acc0, fifo_inst_acc0,
            fifo_X_acc0_slr0, fifo_X_acc0,
            fifo_W_acc0, fifo_W_acc0,
            fifo_from_acc1_to_acc0,
            fifo_acc0_to_sfu,
            fifo_gelu_in,
            fifo_cont_to_acc0,
            fifo_ffn_buffer_out,
            fifo_reduce_acc0,
            fifo_res_acc0
            // fifo_fin
        )
        .invoke<tapa::join>(
            temporal_acc1_slr0,
            seq_len,
            fifo_inst_acc1, fifo_inst_acc1,
            fifo_X_acc1_slr0, fifo_X_acc1,
            fifo_W_acc1, fifo_W_acc1,
            fifo_from_acc1_to_acc0,
            fifo_from_sfu_to_acc1,
            fifo_context,
            fifo_cont_to_acc1,
            fifo_reduce_acc1,
            fifo_res_acc1
            // fifo_fin
        )
        .invoke<tapa::join>(
            residual, seq_len,
            fifo_res_acc0,
            fifo_ln_acc0
        )
        .invoke<tapa::join>(
            residual, seq_len,
            fifo_res_acc1,
            fifo_ln_acc1
        )
        .invoke<tapa::join, NUM_SLR-1>(
            temporal_acc0,
            seq_len,
            fifo_inst_acc0, fifo_inst_acc0,
            fifo_X_acc0, fifo_X_acc0,
            fifo_W_acc0, fifo_W_acc0,
            fifo_from_acc1_to_acc0,
            fifo_acc0_to_sfu,
            fifo_cont_to_acc0,
            fifo_gelu_in,
            fifo_reduce_acc0, fifo_reduce_acc0
        )
        .invoke<tapa::join, NUM_SLR-1>(
            temporal_acc1,
            seq_len,
            fifo_inst_acc1, fifo_inst_acc1,
            fifo_X_acc1, fifo_X_acc1,
            fifo_W_acc1, fifo_W_acc1,
            fifo_from_acc1_to_acc0,
            fifo_from_sfu_to_acc1,
            fifo_context,
            fifo_cont_to_acc1,
            fifo_reduce_acc1, fifo_reduce_acc1
        )
        .invoke<tapa::join>(write_zero, seq_len, fifo_reduce_acc0)
        .invoke<tapa::join>(write_zero, seq_len, fifo_reduce_acc1)
        .invoke<tapa::join, NUM_SLR>(
            sfu_acc_exp, seq_len,
            fifo_acc0_to_sfu,
            fifo_sfu_buf_in
        )
        .invoke<tapa::join>(
            sfu_buffer_slr0, seq_len,
            fifo_sfu_buf_in,
            fifo_ln_acc0,
            fifo_sfu_buf_out
        )
        .invoke<tapa::join>(
            sfu_buffer_slr0, seq_len,
            fifo_sfu_buf_in,
            fifo_ln_acc1,
            fifo_sfu_buf_out
        )
        .invoke<tapa::join, (NUM_SLR-1)*2>(
            sfu_buffer, seq_len,
            fifo_sfu_buf_in,
            fifo_sfu_buf_out
        )
        .invoke<tapa::join>(
            sfu_norm_slr0, seq_len,
            fifo_sfu_buf_out,
            fifo_from_sfu_to_acc1,
            fifo_ffn_buffer_in
        )
        .invoke<tapa::join, NUM_SLR-1>(
            sfu_norm, seq_len,
            fifo_sfu_buf_out,
            fifo_from_sfu_to_acc1
        )
        .invoke<tapa::join>(
            ffn_buffer, seq_len,
            fifo_ffn_buffer_in,
            fifo_ffn_buffer_out
        )
        .invoke<tapa::join, NUM_SLR>(
            context_buffer, seq_len,
            fifo_context,
            fifo_cont_to_acc0, fifo_cont_to_acc1
        )
        // TODO: delete aggreator and write_trash after implementation
        .invoke<tapa::join>(
            write_trash, seq_len,
            fifo_debug
        )
        .invoke<tapa::join, NUM_SLR>(
            aggregator, seq_len,
            fifo_debug,
            fifo_gelu_in,
            fifo_debug
        )
        // .invoke<tapa::join, NUM_SLR>(write_attention, seq_len, acc0_out, fifo_acc0_out)
        .invoke<tapa::join>(write_mtx, L_out, acc0_out, fifo_debug, fifo_fin)
        // .invoke<tapa::join>(write_mtx, L_out, acc1_out, fifo_acc1_out)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count)
        .invoke<tapa::detach>(black_hole_int, fifo_inst_acc0)
        .invoke<tapa::detach>(black_hole_int, fifo_inst_acc1)
        .invoke<tapa::detach>(black_hole_ap_uint_1024, fifo_X_acc0)
        .invoke<tapa::detach>(black_hole_ap_uint_1024, fifo_X_acc1)
        .invoke<tapa::detach>(black_hole_ap_uint_512, fifo_W_acc0)
        .invoke<tapa::detach>(black_hole_ap_uint_512, fifo_W_acc1);
}
