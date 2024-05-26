#include <cmath>
#include <string>
#include <tapa.h>
#include <ap_int.h>

constexpr int D = 1024;
constexpr int D_ffn = 4096;
constexpr int N_head = 16;
constexpr int MAX_SEQ_LEN = 1024;
constexpr int NUM_SLR = 4;
constexpr int D_head = D / N_head;
constexpr int FFN_WEIGHT_SIZE = D * D_ffn;
constexpr int OUT_WEIGHT_SIZE = D * D;
constexpr int QKV_WEIGHT_SIZE = D * D / N_head * NUM_SLR; // multi-head attention

using int_v16 = tapa::vec_t<int, 16>;

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

void read_W(
    const int N,
    tapa::istream<int>& fifo_inst_in,
    tapa::ostream<int>& fifo_inst_out,
    tapa::async_mmap<int_v16>& vec,
    tapa::ostream<int_v16>& fifo_out
){
    int L = fifo_inst_in.read();
    int reload = fifo_inst_in.read();
    fifo_inst_out.write(L);
    fifo_inst_out.write(reload);

    if(reload == 1){
        for(int i_req = 0, i_resp = 0; i_resp < (N >> 4);){
            #pragma HLS pipeline II=1
            if((i_req < (N >> 4)) & !vec.read_addr.full()){
                vec.read_addr.write(i_req);
                i_req++;
            }
            if(!vec.read_data.empty()){
                int_v16 tmp_o; vec.read_data.try_read(tmp_o);
                fifo_out.write(tmp_o);
                i_resp++;
            }
        }
    }
}

void read_X(
    const int N,
    tapa::async_mmap<int>& vec,
    tapa::ostream<int>& fifo_out
){
    for(int i_req = 0, i_resp = 0; i_resp < N;){
        #pragma HLS pipeline II=1
        if((i_req < N) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        if(!vec.read_data.empty()){
            int tmp_o; vec.read_data.try_read(tmp_o);
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
    tapa::async_mmap<int>& output_mtx,
    tapa::istream<int>& fifo_in,
    tapa::ostream<bool>& fifo_fin
){
    for(int i_req = 0, i_resp = 0; i_resp < N;){
        #pragma HLS pipeline II=1
        if((i_req < N) & !fifo_in.empty() & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
            output_mtx.write_addr.try_write(i_req);
            int tmp; fifo_in.try_read(tmp);
            output_mtx.write_data.try_write(tmp);
            ++i_req;
        }
        if(!output_mtx.write_resp.empty()){
            i_resp += unsigned(output_mtx.write_resp.read(nullptr))+1;
        }
    }
    fifo_fin.write(true);
}

void temporal_acc0(
    const int slr_idx,
    tapa::istream<int>& fifo_len_in,
    tapa::ostream<int>& fifo_len_out,
    tapa::istream<int>& fifo_X_in,
    tapa::ostream<int>& fifo_X_out,
    tapa::istream<int_v16>& fifo_W_in,
    tapa::ostream<int_v16>& fifo_W_out,
    tapa::ostream<int>& fifo_O_out
){

    int scratchpad[MAX_SEQ_LEN][D_head];
    int W_q[D][D_head];

    const int L = fifo_len_in.read();
    const int reload = fifo_len_in.read();
    fifo_len_out.write(L);
    fifo_len_out.write(reload);

    // load weights and forward
    if(reload == 1) {
        for(int i = 0; i < D; i++){
            for(int j = 0; j < (D_head >> 2);){
                if(!fifo_W_in.empty() & !fifo_W_out.full()){
                    int_v16 val; fifo_W_in.try_read(val);
                    for(int k = 0; k < 4; k++){
                        W_q[i][j*4 + k] = val[slr_idx*4+k];
                    }
                    fifo_W_out.try_write(val);
                    j++;
                }
            }
        }
    }
    
    // stage 1: compute Q 
    for(int i = 0; i < L; i++){

        int X[D];

        for(int j = 0; j < D;){
            if(!fifo_X_in.empty() & !fifo_X_out.full()){
                int val; fifo_X_in.try_read(val);
                fifo_X_out.try_write(val);
                X[j] = val;
                j++;
            }
        }

        for(int j = 0; j < D_head; j++){
            int acc = 0;
            for(int k = 0; k < D; k++){
                acc += X[k] * W_q[k][j];
            }
            scratchpad[i][j] = acc;
        }
    }

    // write out for debug
    for(int i = 0; i < L; i++){
        for(int j = 0; j < D_head; j++){
            fifo_O_out.write(scratchpad[i][j]);
        }
    }
}

void temporal_acc1(
    const int slr_idx,
    tapa::istream<int>& fifo_len_in,
    tapa::ostream<int>& fifo_len_out,
    tapa::istream<int>& fifo_X_in,
    tapa::ostream<int>& fifo_X_out,
    tapa::istream<int_v16>& fifo_W_in,
    tapa::ostream<int_v16>& fifo_W_out,
    tapa::ostream<int>& fifo_O_out
){

    int scratchpad[MAX_SEQ_LEN][D_head];
    int W_v[D][D_head];

    const int L = fifo_len_in.read();
    const int reload = fifo_len_in.read();
    fifo_len_out.write(L);
    fifo_len_out.write(reload);

    // load weights and forward
    if(reload == 1) {
        for(int i = 0; i < D; i++){
            for(int j = 0; j < (D_head >> 2);){
                if(!fifo_W_in.empty() & !fifo_W_out.full()){
                    int_v16 val; fifo_W_in.try_read(val);
                    for(int k = 0; k < 4; k++){
                        W_v[i][j*4 + k] = val[slr_idx*4+k];
                    }
                    fifo_W_out.try_write(val);
                    j++;
                }
            }
        }
    }
    
    // stage 1: compute V 
    for(int i = 0; i < L; i++){

        int X[D];

        for(int j = 0; j < D;){
            if(!fifo_X_in.empty() & !fifo_X_out.full()){
                int val; fifo_X_in.try_read(val);
                fifo_X_out.try_write(val);
                X[j] = val;
                j++;
            }
        }

        for(int j = 0; j < D_head; j++){
            int acc = 0;
            for(int k = 0; k < D; k++){
                acc += X[k] * W_v[k][j];
            }
            scratchpad[i][j] = acc;
        }

    }

    // write out for debug
    for(int i = 0; i < L; i++){
        for(int j = 0; j < D_head; j++){
            fifo_O_out.write(scratchpad[i][j]);
        }
    }
}




void measure_cycle(tapa::istreams<bool, 8>& fifo_fin, tapa::mmap<int> cycle_count){
    for(int cycle = 0, count = 0;;cycle++){
        bool flag_cont = false;
        for(int i = 0; i < 8; i++){
            flag_cont |= fifo_fin[i].empty();
        }
        if(!flag_cont){
            for(int i = 0; i < 8; i++){
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
    tapa::mmap<int> X_acc0,
    tapa::mmap<int> X_acc1,
    tapa::mmap<int_v16> W_acc0,
    tapa::mmap<int_v16> W_acc1,
    tapa::mmaps<int, NUM_SLR> acc0_out,
    tapa::mmaps<int, NUM_SLR> acc1_out,
    tapa::mmap<int> cycle_count
){
    tapa::streams<int, NUM_SLR+2> fifo_inst_acc0("fifo_inst_acc0");
    tapa::streams<int, NUM_SLR+2> fifo_inst_acc1("fifo_inst_acc1");
    tapa::streams<int, NUM_SLR+1> fifo_X_acc0("fifo_X_acc0");
    tapa::streams<int, NUM_SLR+1> fifo_X_acc1("fifo_X_acc1");
    tapa::streams<int_v16, NUM_SLR+1> fifo_W_acc0("fifo_W_acc0");
    tapa::streams<int_v16, NUM_SLR+1> fifo_W_acc1("fifo_W_acc1");
    tapa::streams<int, NUM_SLR> fifo_acc0_out("fifo_acc0_out");
    tapa::streams<int, NUM_SLR> fifo_acc1_out("fifo_acc1_out");
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
            fifo_acc0_out
        )
        .invoke<tapa::join, NUM_SLR>(
            temporal_acc1,
            tapa::seq(),
            fifo_inst_acc1, fifo_inst_acc1,
            fifo_X_acc1, fifo_X_acc1,
            fifo_W_acc1, fifo_W_acc1,
            fifo_acc1_out
        )
        .invoke<tapa::join, NUM_SLR>(write_mtx, L_out, acc0_out, fifo_acc0_out, fifo_fin)
        .invoke<tapa::join, NUM_SLR>(write_mtx, L_out, acc1_out, fifo_acc1_out, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count)
        .invoke<tapa::detach>(black_hole_int, fifo_inst_acc0)
        .invoke<tapa::detach>(black_hole_int, fifo_inst_acc1)
        .invoke<tapa::detach>(black_hole_int, fifo_X_acc0)
        .invoke<tapa::detach>(black_hole_int, fifo_X_acc1)
        .invoke<tapa::detach>(black_hole_int_v16, fifo_W_acc0)
        .invoke<tapa::detach>(black_hole_int_v16, fifo_W_acc1);
}