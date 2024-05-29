#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>
#include <tapa.h>
#include <gflags/gflags.h>
#include <ap_int.h>

constexpr int D = 1024;
constexpr int D_ffn = 4096;
constexpr int N_head = 16;
constexpr int MAX_SEQ_LEN = 1024;
constexpr int NUM_SLR = 3;
constexpr int NUM_DUM_SLR = 4;
constexpr int D_head = D / N_head;
constexpr int FFN_WEIGHT_SIZE = D * D_ffn;
constexpr int OUT_WEIGHT_SIZE = D * D;
constexpr int QKV_WEIGHT_SIZE = D * D / N_head * NUM_SLR; // multi-head attention

using std::vector;
using int_v16 = tapa::vec_t<int, 16>;
using int4_v128 = tapa::vec_t<ap_int<4>, 128>;
using int8_v64 = tapa::vec_t<ap_int<8>, 64>;

void opt_kernel(
    const int L,
    const int L_out,
    tapa::mmap<int> inst, // inst[0] = L, inst[1] = reload_weight
    tapa::mmap<int8_v64> X_acc0,
    tapa::mmap<int8_v64> X_acc1,
    tapa::mmap<int8_v64> W_acc0,
    tapa::mmap<int8_v64> W_acc1,
    tapa::mmaps<int, NUM_SLR> acc0_out,
    tapa::mmaps<int, NUM_SLR> acc1_out,
    tapa::mmap<int> cycle_count
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int L = argc > 1 ? atoll(argv[1]) : MAX_SEQ_LEN;

    srand((unsigned)time(nullptr));

    // data preparation
    aligned_vector<int> inst = {L, 1};
    aligned_vector<ap_int<8>> X_acc0(L * D);
    aligned_vector<ap_int<8>> X_acc1(L * D);
    aligned_vector<ap_int<8>> W_acc0(D * D_head * NUM_DUM_SLR/2);
    aligned_vector<ap_int<8>> W_acc1(D * D_head * NUM_DUM_SLR/2);
    vector<aligned_vector<int>> acc0_out(NUM_SLR, aligned_vector<int>(L * D_head));
    vector<aligned_vector<int>> acc1_out(NUM_SLR, aligned_vector<int>(L * D_head));
    aligned_vector<int> cycle_count(1);


    vector<int> X_copy(L * D);
    vector<vector<int>> W_acc0_split(NUM_DUM_SLR, vector<int>(D * D_head));
    vector<vector<int>> W_acc1_split(NUM_DUM_SLR, vector<int>(D * D_head));
    vector<aligned_vector<int>> acc0_out_golden(NUM_DUM_SLR, aligned_vector<int>(L * D_head));
    vector<aligned_vector<int>> acc1_out_golden(NUM_DUM_SLR, aligned_vector<int>(L * D_head));

    for(int i = 0; i < L * D; i++){
        int val = (rand() % 64) + 1;
        ap_int<32> full = tapa::bit_cast<ap_int<32>>(val);
        X_copy[i] = val;
        X_acc0[i] = ap_int<8>(full(7, 0));
        X_acc1[i] = ap_int<8>(full(7, 0));
    }

    for(int i = 0; i < D * D_head * NUM_DUM_SLR; i++){
        int val = (rand() % 6) - 1;
        ap_int<32> full = tapa::bit_cast<ap_int<32>>(val);
        W_acc0[i/2]((i%2+1)*4-1, (i%2)*4) = ap_int<4>(full(3, 0));
        W_acc0_split[(i / 32) % 4][(i / 128) * 32 + (i % 32)] = val;
    }

    for(int i = 0; i < D * D_head * NUM_DUM_SLR; i++){
        int val = (rand() % 6) - 1;
        ap_int<32> full = tapa::bit_cast<ap_int<32>>(val);
        W_acc1[i/2]((i%2+1)*4-1, (i%2)*4) = ap_int<4>(full(3, 0));
        W_acc1_split[(i / 32) % 4][(i / 128) * 32 + (i % 32)] = val;
    }

    // cpu 
    for(int i = 0; i < NUM_SLR; i++){
        //WqX
        for(int j = 0; j < L; j++){
            for(int k = 0; k < D_head; k++){
                int acc = 0;
                for(int l = 0; l < D; l++){
                    acc += X_copy[j*D+l] * W_acc0_split[i][l*D_head + k];
                }
                acc0_out_golden[i][j * D_head + k] = acc;
            }
        }

        //WvX
        for(int j = 0; j < L; j++){
            for(int k = 0; k < D_head; k++){
                int acc = 0;
                for(int l = 0; l < D; l++){
                    acc += X_copy[j*D+l] * W_acc1_split[i][l*D_head + k];
                }
                acc1_out_golden[i][j * D_head + k] = acc;
            }
        }
    }


    // invoke the kernel

    int64_t kernel_time_ns = tapa::invoke(opt_kernel, FLAGS_bitstream,
        L * D, L * D_head,
        tapa::read_only_mmap<int>(inst), 
        tapa::read_only_mmap<ap_int<8>>(X_acc0).reinterpret<int8_v64>(), 
        tapa::read_only_mmap<ap_int<8>>(X_acc1).reinterpret<int8_v64>(), 
        tapa::read_only_mmap<ap_int<8>>(W_acc0).reinterpret<int8_v64>(), 
        tapa::read_only_mmap<ap_int<8>>(W_acc1).reinterpret<int8_v64>(), 
        tapa::write_only_mmaps<int, NUM_SLR>(acc0_out), 
        tapa::write_only_mmaps<int, NUM_SLR>(acc1_out), 
        tapa::write_only_mmap<int>(cycle_count));
    
    std::clog << "cycle time: " << cycle_count[0] << std::endl;
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;

    int error = 0;

    // compare
    for(int i = 0; i < NUM_SLR; i++){
        for(int j = 0; j < L * D_head; j++){
            if(acc1_out[i][j] != acc1_out_golden[i][j]){
                 std::clog << "slr: " << i << ", index: " << j << ", actual: " << acc0_out[i][j] << ", expect: " << acc0_out_golden[i][j] << std::endl;
                 error++;
            }
        }
    }

    if (error == 0) {
        std::clog << "PASSED" << std::endl;
    } else {
        std::clog << "FAILED" << std::endl;
        return 1;
    }
    return 0;
        
}

