#include <cmath>
#include <string>
#include <tapa.h>
#include <ap_int.h>

constexpr int D = 1024;
constexpr int D_ffn = 4096;
constexpr int N_head = 16;
constexpr int MAX_SEQ_LEN = 1024;
constexpr int NUM_SLR = 3;
constexpr int NUM_DUM_SLR = 4;
constexpr int TOTAL_PORT = NUM_SLR * 2;
constexpr int D_head = D / N_head;
constexpr int D_head_div_16 = D_head / 16;
constexpr int D_head_div_8 = D_head / 8;
constexpr int D_head_div_4 = D_head / 4;
constexpr int D_div_8 = D / 8;
constexpr int FFN_WEIGHT_SIZE = D * D_ffn;
constexpr int OUT_WEIGHT_SIZE = D * D;
constexpr int WEIGHT_D = D * 2;
constexpr int QKV_WEIGHT_SIZE = D * D / N_head * NUM_DUM_SLR; // multi-head attention

using int_v16 = tapa::vec_t<int, 16>;
using int4_v128 = tapa::vec_t<ap_int<4>, 128>;
using int8_v64 = tapa::vec_t<ap_int<8>, 64>;

template <typename data_t>
inline void bh(tapa::istream<data_t> & q) {
#pragma HLS inline
    for (;;) {
#pragma HLS pipeline II=1
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

void read_W(
    const int N,
    tapa::istream<int>& fifo_inst_in,
    tapa::ostream<int>& fifo_inst_out,
    tapa::async_mmap<ap_uint<512>>& vec,
    tapa::ostream<ap_uint<512>>& fifo_out
){
    int L = fifo_inst_in.read();
    int reload = fifo_inst_in.read();
    fifo_inst_out.write(L);
    fifo_inst_out.write(reload);

    if(reload == 1){
        for(int i_req = 0, i_resp = 0; i_resp < (N >> 7);){
            #pragma HLS pipeline II=1
            if((i_req < (N >> 7)) & !vec.read_addr.full()){
                vec.read_addr.write(i_req);
                i_req++;
            }
            if(!vec.read_data.empty()){
                ap_uint<512> tmp_o; vec.read_data.try_read(tmp_o);
                fifo_out.write(tmp_o);
                i_resp++;
            }
        }
    }
}

void read_X(
    const int N,
    tapa::async_mmap<ap_uint<512>>& vec,
    tapa::ostream<ap_uint<512>>& fifo_out
){
    for(int i_req = 0, i_resp = 0; i_resp < (N >> 6);){
        #pragma HLS pipeline II=1
        if((i_req < (N >> 6)) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        if(!vec.read_data.empty()){
            ap_uint<512> tmp_o; vec.read_data.try_read(tmp_o);
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

void read_inst(
    const int N,
    tapa::async_mmap<int>& vec,
    tapa::ostream<int>& fifo_out_acc0,
    tapa::ostream<int>& fifo_out_acc1
){
    for(int i_req = 0, i_resp = 0; i_resp < N;){
        #pragma HLS pipeline II=1
        if((i_req < N) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        if(!vec.read_data.empty()){
            int tmp_o; vec.read_data.try_read(tmp_o);
            fifo_out_acc0.write(tmp_o);
            fifo_out_acc1.write(tmp_o);
            i_resp++;
        }
    }
}

void write_mtx(
    const int N,
    tapa::async_mmap<ap_uint<64>>& output_mtx,
    tapa::istreams<ap_uint<64>, NUM_SLR>& fifo_in
){

    for(int n = 0; n < NUM_SLR; n++){
        int offset = n * N;
        for(int i_req = 0, i_resp = 0; i_resp < N;){
            #pragma HLS pipeline II=1
            if((i_req < N) & !fifo_in[n].empty() & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
                output_mtx.write_addr.try_write(i_req + offset);
                ap_uint<64> tmp; fifo_in[n].try_read(tmp);
                output_mtx.write_data.try_write(tmp);
                ++i_req;
            }
            if(!output_mtx.write_resp.empty()){
                i_resp += unsigned(output_mtx.write_resp.read(nullptr))+1;
            }
        }
    }
}

// acc slr0 master node
void temporal_acc0_slr0(
    tapa::istream<int>& fifo_len_in,
    tapa::ostream<int>& fifo_len_out,
    tapa::istream<int8_v64>& fifo_X_in,
    tapa::ostream<int8_v64>& fifo_X_out, // 8-bit activation
    tapa::istream<int8_v64>& fifo_W_in,
    tapa::ostream<int8_v64>& fifo_W_out, // 4-bit weight
    tapa::ostream<ap_uint<64>>& fifo_O_out,
    tapa::ostream<bool>& fifo_fin
){

    ap_uint<64> scratchpad[MAX_SEQ_LEN][D_head_div_8]; // 8 bit, store intermediate results
    ap_uint<64> W_q[D][D_head_div_16]; // 4 bit, store all weights

    #pragma HLS array_partition variable=W_q cyclic dim=1 factor=4
    #pragma HLS array_partition variable=W_q dim=2 complete
    #pragma HLS array_partition variable=scratchpad cyclic dim=1 factor=2
    #pragma HLS array_partition variable=scratchpad dim=2 complete
    #pragma HLS bind_storage variable=W_q type=ram_2p impl=uram

    ap_uint<64> X[MAX_SEQ_LEN][D_div_8]; // 8 bit
    #pragma HLS array_partition variable=X cyclic dim=2 factor=8
    #pragma HLS array_partition variable=X cyclic dim=1 factor=4

    const int L = fifo_len_in.read();
    const int reload = fifo_len_in.read();
    fifo_len_out.write(L);
    fifo_len_out.write(reload);

    // load weights and forward
    if(reload == 1) {
        for(int i = 0; i < D; i++){
            load_weight:
            for(int j = 0; j < (D_head_div_16 >> 1);){
                if(!fifo_W_in.empty()){
                    int8_v64 val; fifo_W_in.try_read(val);

                    for(int k = 0; k < 2; k++){
                        #pragma HLS unroll
                        ap_uint<64> tmp = 0;
                        int offset_k = (k << 3);

                        for(int m = 0; m < 8; m++){
                            #pragma HLS unroll
                            tmp(m*8+7, m*8) = tapa::bit_cast<ap_uint<8>>(val[offset_k+m]);
                        }

                        W_q[i][j*2 + k] = tmp;
                    }
                    fifo_W_out.write(val);
                    j++;
                }
            }
        }
    }
    
    // stage 1: compute Q 
    for(int i = 0; i < (L >> 2); i++){ // make sure L is multiple of 4 & larger than 16
        #pragma HLS dataflow
        // ap_uint<64> X[4][D_div_8];
        // #pragma HLS array_partition variable=X cyclic dim=2 factor=8
        // #pragma HLS array_partition variable=X complete dim=1

        for(int ii = 0; ii < 4; ii++){
load_x:
            for(int j = 0; j < (D_div_8 >> 3);){
                if(!fifo_X_in.empty()){
                    int8_v64 val; fifo_X_in.try_read(val);
                    fifo_X_out.write(val);
                    
                    for(int k = 0; k < 8; k++){
                        #pragma HLS unroll
                        ap_uint<64> tmp = 0;

                        for(int m = 0; m < 8; m++){
                            #pragma HLS unroll
                            tmp(m*8+7, m*8) = tapa::bit_cast<ap_uint<8>>(val[k*8+m]);
                        }

                        X[i*4+ii][(j << 3) + k] = tmp;
                    }
                    j++;
                }
            }
        }

        int acc_vec[4][8][D_head];
        #pragma HLS array_partition variable=acc_vec dim=1 complete
        #pragma HLS array_partition variable=acc_vec dim=2 complete
        #pragma HLS array_partition variable=acc_vec dim=3 complete

        for(int ii = 0; ii < 4; ii++){
            #pragma HLS unroll
            for(int kk = 0; kk < 8; kk++){
                #pragma HLS unroll
                for(int k = 0; k < D_head; k++){
                    #pragma HLS unroll
                    acc_vec[ii][kk][k] = 0;
                }
            }
        }

compute:
        for(int k = 0; k < D_div_8; k++){
            #pragma HLS pipeline II=1
            for(int ii = 0; ii < 4; ii++){
                #pragma HLS unroll
                ap_uint<64> x_vec = X[i*4+ii][k];
                for(int kk = 0; kk < 8; kk++){
                    #pragma HLS unroll
                    for(int l = 0; l < D_head_div_16; l++){
                        #pragma HLS unroll
                        ap_uint<64> w_vec = W_q[(k << 3) + kk][l];
                        for(int m = 0; m < 8; m++){ // dsp packing 8 * (4+4)
                            #pragma HLS unroll
                            ap_uint<8> w_2ele = ap_uint<8>(w_vec((m+1)*8-1, m*8));
                            ap_int<16> w_pack = ap_int<16>((w_2ele(7,4), ap_uint<12>(0))) + ap_int<4>(w_2ele(3,0));
                            ap_int<12> res0;
                            ap_int<12> res1;
                            (res1, res0) = ap_int<8>(x_vec((kk+1)*8-1, kk*8)) * w_pack;
                            res1 = res1 + res0[11];
                            acc_vec[ii][kk][l*16+m*2] += res0;
                            acc_vec[ii][kk][l*16+m*2+1] += res1;
                        }
                    }
                }
            }
        }

reduction:
        for(int kk = 1; kk < 8; kk++){
            for(int ii = 0; ii < 4; ii++){
                #pragma HLS unroll
                for(int k = 0; k < D_head; k++){
                    #pragma HLS unroll
                    acc_vec[ii][0][k] += acc_vec[ii][kk][k];
                }
            }
        }

scale_and_pack:
        for(int ii = 0; ii < 4; ii++){
            #pragma HLS unroll
            for(int k = 0; k < D_head_div_8; k++){
                #pragma HLS unroll
                ap_uint<64> pack;
                for(int m = 0; m < 8; m++){
                    pack(m*8+7, m*8) = ap_int<8>(std::min(std::max(acc_vec[ii][0][k*8+m] >> 8, -128), 127)); // rescale & clamp
                }
                scratchpad[i*4+ii][k] = pack;
            }
        }
    }
    fifo_fin.write(true);

    // write out for debug
write:
    for(int i = 0; i < L; i++){
        for(int j = 0; j < D_head_div_8; j++){
            #pragma HLS pipeline II=1
            fifo_O_out.write(scratchpad[i][j]);
        }
    }
}

void temporal_acc0(
    const int slr_idx,
    tapa::istream<int>& fifo_len_in,
    tapa::ostream<int>& fifo_len_out,
    tapa::istream<ap_uint<512>>& fifo_X_in,
    tapa::ostream<ap_uint<512>>& fifo_X_out, // 8-bit activation
    tapa::istream<ap_uint<512>>& fifo_W_in,
    tapa::ostream<ap_uint<512>>& fifo_W_out, // 4-bit weight
    tapa::ostream<ap_uint<64>>& fifo_O_out,
    tapa::ostream<bool>& fifo_fin
){

    ap_uint<64> scratchpad[MAX_SEQ_LEN][D_head_div_8]; 
    ap_uint<64> W_q[D][D_head_div_16]; // 4 bit

    #pragma HLS array_partition variable=W_q cyclic dim=1 factor=4
    #pragma HLS array_partition variable=W_q dim=2 complete
    #pragma HLS array_partition variable=scratchpad cyclic dim=1 factor=2
    #pragma HLS array_partition variable=scratchpad dim=2 complete
    #pragma HLS bind_storage variable=W_q type=ram_2p impl=uram

    const int L = fifo_len_in.read();
    const int reload = fifo_len_in.read();
    fifo_len_out.write(L);
    fifo_len_out.write(reload);

    // load weights and forward
    if(reload == 1) {
        for(int i = 0; i < D; i++){
            load_weight:
            for(int j = 0; j < (D_head_div_16 >> 1);){
                if(!fifo_W_in.empty()){
                    ap_uint<512> val; fifo_W_in.try_read(val);

                    for(int k = 0; k < 2; k++){
                        #pragma HLS unroll
                        W_q[i][j*2 + k] = ap_uint<64>(val((slr_idx*2+k)*64+63, (slr_idx*2+k)*64));
                    }
                    fifo_W_out.write(val);
                    j++;
                }
            }
        }
    }
    
    // stage 1: compute Q 
    for(int i = 0; i < (L >> 2); i++){ // make sure L is multiple of 4

        ap_uint<64> X[4][D_div_8];
        #pragma HLS array_partition variable=X cyclic dim=2 factor=8
        #pragma HLS array_partition variable=X complete dim=1

        for(int ii = 0; ii < 4; ii++){
load_x:
            for(int j = 0; j < (D_div_8 >> 3);){
                if(!fifo_X_in.empty()){
                    ap_uint<512> val; fifo_X_in.try_read(val);
                    fifo_X_out.write(val);
                    
                    for(int k = 0; k < 8; k++){
                        #pragma HLS unroll
                        X[ii][(j << 3) + k] = ap_uint<64>(val(k*64+63, k*64));
                    }
                    j++;
                }
            }
        }

        int acc_vec[4][8][D_head];
        #pragma HLS array_partition variable=acc_vec dim=1 complete
        #pragma HLS array_partition variable=acc_vec dim=2 complete
        #pragma HLS array_partition variable=acc_vec dim=3 complete

        for(int ii = 0; ii < 4; ii++){
            #pragma HLS unroll
            for(int kk = 0; kk < 8; kk++){
                #pragma HLS unroll
                for(int k = 0; k < D_head; k++){
                    #pragma HLS unroll
                    acc_vec[ii][kk][k] = 0;
                }
            }
        }

compute:
        for(int k = 0; k < D_div_8; k++){
            #pragma HLS pipeline II=1
            for(int ii = 0; ii < 4; ii++){
                #pragma HLS unroll
                ap_uint<64> x_vec = X[ii][k];
                for(int kk = 0; kk < 8; kk++){
                    #pragma HLS unroll
                    for(int l = 0; l < D_head_div_16; l++){
                        #pragma HLS unroll
                        ap_uint<64> w_vec = W_q[(k << 3) + kk][l];
                        for(int m = 0; m < 8; m++){ // dsp packing 8 * (4+4)
                            #pragma HLS unroll
                            ap_uint<8> w_2ele = ap_uint<8>(w_vec((m+1)*8-1, m*8));
                            ap_int<16> w_pack = ap_int<16>((w_2ele(7,4), ap_uint<12>(0))) + ap_int<4>(w_2ele(3,0));
                            ap_int<12> res0;
                            ap_int<12> res1;
                            (res1, res0) = ap_int<8>(x_vec((kk+1)*8-1, kk*8)) * w_pack;
                            res1 = res1 + res0[11];
                            acc_vec[ii][kk][l*16+m*2] += res0;
                            acc_vec[ii][kk][l*16+m*2+1] += res1;
                        }
                    }
                }
            }
        }

reduction:
        for(int kk = 1; kk < 8; kk++){
            for(int ii = 0; ii < 4; ii++){
                #pragma HLS unroll
                for(int k = 0; k < D_head; k++){
                    #pragma HLS unroll
                    int offset = k%8;
                    acc_vec[ii][0][k] += acc_vec[ii][kk][k];
                    if(kk == 7) scratchpad[i*4+ii][k/8](offset*8+7, offset*8) = ap_int<8>(std::min(std::max(acc_vec[ii][0][k] >> 8, -128), 127)); // rescale & clamp
                }
            }
        }
    }
    fifo_fin.write(true);

    // write out for debug
write:
    for(int i = 0; i < L; i++){
        for(int j = 0; j < D_head_div_8; j++){
            #pragma HLS pipeline II=1
            fifo_O_out.write(scratchpad[i][j]);
        }
    }
}

void temporal_acc1(
    const int slr_idx,
    tapa::istream<int>& fifo_len_in,
    tapa::ostream<int>& fifo_len_out,
    tapa::istream<ap_uint<512>>& fifo_X_in,
    tapa::ostream<ap_uint<512>>& fifo_X_out, // 8-bit activation
    tapa::istream<ap_uint<512>>& fifo_W_in,
    tapa::ostream<ap_uint<512>>& fifo_W_out, // 4-bit weight
    tapa::ostream<ap_uint<64>>& fifo_O_out,
    tapa::ostream<bool>& fifo_fin
){

    ap_uint<64> scratchpad[MAX_SEQ_LEN][D_head_div_8];
    ap_uint<64> W_v[D][D_head_div_16];

    #pragma HLS array_partition variable=W_v cyclic dim=1 factor=4
    #pragma HLS array_partition variable=W_v dim=2 complete
    #pragma HLS array_partition variable=scratchpad cyclic dim=1 factor=2
    #pragma HLS array_partition variable=scratchpad dim=2 complete
    #pragma HLS bind_storage variable=W_v type=ram_2p impl=uram

    const int L = fifo_len_in.read();
    const int reload = fifo_len_in.read();
    fifo_len_out.write(L);
    fifo_len_out.write(reload);

    // load weights and forward
    if(reload == 1) {

        for(int i = 0; i < D; i++){
            load_weight:
            for(int j = 0; j < (D_head_div_16 >> 1);){
                if(!fifo_W_in.empty()){
                    ap_uint<512> val; fifo_W_in.try_read(val);

                    for(int k = 0; k < 2; k++){
                        #pragma HLS unroll
                        W_v[i][j*2 + k] = ap_uint<64>(val((slr_idx*2+k)*64+63, (slr_idx*2+k)*64));
                    }
                    fifo_W_out.write(val);
                    j++;
                }
            }
        }
    }
    
    // stage 1: compute Q 
    for(int i = 0; i < (L >> 2); i++){ // make sure L is multiple of 4

        ap_uint<64> X[4][D_div_8];
        #pragma HLS array_partition variable=X cyclic dim=2 factor=8
        #pragma HLS array_partition variable=X complete dim=1

        for(int ii = 0; ii < 4; ii++){
load_x:
            for(int j = 0; j < (D_div_8 >> 3);){
                if(!fifo_X_in.empty()){
                    ap_uint<512> val; fifo_X_in.try_read(val);
                    fifo_X_out.write(val);
                    
                    for(int k = 0; k < 8; k++){
                        #pragma HLS unroll
                        X[ii][(j << 3) + k] = ap_uint<64>(val(k*64+63, k*64));
                    }
                    j++;
                }
            }
        }

        int acc_vec[4][8][D_head];
        #pragma HLS array_partition variable=acc_vec dim=1 complete
        #pragma HLS array_partition variable=acc_vec dim=2 complete
        #pragma HLS array_partition variable=acc_vec dim=3 complete

        for(int ii = 0; ii < 4; ii++){
            #pragma HLS unroll
            for(int kk = 0; kk < 8; kk++){
                #pragma HLS unroll
                for(int k = 0; k < D_head; k++){
                    #pragma HLS unroll
                    acc_vec[ii][kk][k] = 0;
                }
            }
        }

compute:
        for(int k = 0; k < D_div_8; k++){
            #pragma HLS pipeline II=1
            for(int ii = 0; ii < 4; ii++){
                #pragma HLS unroll
                ap_uint<64> x_vec = X[ii][k];
                for(int kk = 0; kk < 8; kk++){
                    #pragma HLS unroll
                    for(int l = 0; l < D_head_div_16; l++){
                        #pragma HLS unroll
                        ap_uint<64> w_vec = W_v[(k << 3) + kk][l];
                        for(int m = 0; m < 8; m++){ // dsp packing 8 * (4+4)
                            #pragma HLS unroll
                            ap_uint<8> w_2ele = ap_uint<8>(w_vec((m+1)*8-1, m*8));
                            ap_int<16> w_pack = ap_int<16>((w_2ele(7,4), ap_uint<12>(0))) + ap_int<4>(w_2ele(3,0));
                            ap_int<12> res0;
                            ap_int<12> res1;
                            (res1, res0) = ap_int<8>(x_vec((kk+1)*8-1, kk*8)) * w_pack;
                            res1 = res1 + res0[11];
                            acc_vec[ii][kk][l*16+m*2] += res0;
                            acc_vec[ii][kk][l*16+m*2+1] += res1;
                        }
                    }
                }
            }
        }

reduction:
        for(int kk = 1; kk < 8; kk++){
            for(int ii = 0; ii < 4; ii++){
                #pragma HLS unroll
                for(int k = 0; k < D_head; k++){
                    #pragma HLS unroll
                    int offset = k%8;
                    acc_vec[ii][0][k] += acc_vec[ii][kk][k];
                    if(kk == 7) scratchpad[i*4+ii][k/8](offset*8+7, offset*8) = ap_int<8>(std::min(std::max(acc_vec[ii][0][k] >> 8, -128), 127)); // rescale & clamp
                }
            }
        }
    }
    fifo_fin.write(true);

    // write out for debug
write:
    for(int i = 0; i < L; i++){
        for(int j = 0; j < D_head_div_8; j++){
            #pragma HLS pipeline II=1
            fifo_O_out.write(scratchpad[i][j]);
        }
    }
}


void measure_cycle(tapa::istreams<bool, TOTAL_PORT>& fifo_fin, tapa::mmap<int> cycle_count){
    for(int cycle = 0;;cycle++){
        bool flag_cont = false;
        for(int i = 0; i < TOTAL_PORT; i++){
            flag_cont |= fifo_fin[i].empty();
        }
        if(!flag_cont){
            for(int i = 0; i < TOTAL_PORT; i++){
                fifo_fin[i].read(nullptr);
            }
            cycle_count[0] = cycle;
            break;
        }
    }
}

void opt_kernel(
    const int L,
    const int L_out,
    tapa::mmap<int> inst, // inst[0] = L, inst[1] = reload_weight
    tapa::mmap<ap_uint<512>> X_acc0,
    tapa::mmap<ap_uint<512>> X_acc1,
    tapa::mmap<ap_uint<512>> W_acc0,
    tapa::mmap<ap_uint<512>> W_acc1,
    tapa::mmap<ap_uint<64>> acc0_out,
    tapa::mmap<ap_uint<64>> acc1_out,
    tapa::mmap<int> cycle_count
){
    tapa::streams<int, NUM_SLR+2> fifo_inst_acc0("fifo_inst_acc0");
    tapa::streams<int, NUM_SLR+2> fifo_inst_acc1("fifo_inst_acc1");
    tapa::streams<ap_uint<512>, NUM_SLR+1, 16> fifo_X_acc0("fifo_X_acc0");
    tapa::streams<ap_uint<512>, NUM_SLR+1, 16> fifo_X_acc1("fifo_X_acc1");
    tapa::streams<ap_uint<512>, NUM_SLR+1> fifo_W_acc0("fifo_W_acc0");
    tapa::streams<ap_uint<512>, NUM_SLR+1> fifo_W_acc1("fifo_W_acc1");
    tapa::streams<ap_uint<64>, NUM_SLR> fifo_acc0_out("fifo_acc0_out");
    tapa::streams<ap_uint<64>, NUM_SLR> fifo_acc1_out("fifo_acc1_out");
    tapa::streams<bool, NUM_SLR*2> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(read_inst, 2, inst, fifo_inst_acc0, fifo_inst_acc1)
        .invoke<tapa::join>(read_W, QKV_WEIGHT_SIZE, fifo_inst_acc0, fifo_inst_acc0, W_acc0, fifo_W_acc0)
        .invoke<tapa::join>(read_W, QKV_WEIGHT_SIZE, fifo_inst_acc1, fifo_inst_acc1, W_acc1, fifo_W_acc1)
        .invoke<tapa::join>(read_X, L, X_acc0, fifo_X_acc0)
        .invoke<tapa::join>(read_X, L, X_acc1, fifo_X_acc1)
        .invoke<tapa::join, NUM_SLR>(
            temporal_acc0,
            tapa::seq(),
            fifo_inst_acc0, fifo_inst_acc0,
            fifo_X_acc0, fifo_X_acc0,
            fifo_W_acc0, fifo_W_acc0,
            fifo_acc0_out,
            fifo_fin
        )
        .invoke<tapa::join, NUM_SLR>(
            temporal_acc1,
            tapa::seq(),
            fifo_inst_acc1, fifo_inst_acc1,
            fifo_X_acc1, fifo_X_acc1,
            fifo_W_acc1, fifo_W_acc1,
            fifo_acc1_out,
            fifo_fin
        )
        .invoke<tapa::join>(write_mtx, L_out, acc0_out, fifo_acc0_out)
        .invoke<tapa::join>(write_mtx, L_out, acc1_out, fifo_acc1_out)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count)
        .invoke<tapa::detach>(black_hole_int, fifo_inst_acc0)
        .invoke<tapa::detach>(black_hole_int, fifo_inst_acc1)
        .invoke<tapa::detach>(black_hole_ap_uint_512, fifo_X_acc0)
        .invoke<tapa::detach>(black_hole_ap_uint_512, fifo_X_acc1)
        .invoke<tapa::detach>(black_hole_ap_uint_512, fifo_W_acc0)
        .invoke<tapa::detach>(black_hole_ap_uint_512, fifo_W_acc1);
}