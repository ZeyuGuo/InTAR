/*
FIXME: This host is from intrra host. 
*/

#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>
#include <cmath>
#include <tapa.h>
#include <gflags/gflags.h>
#include <ap_int.h>

#define N 256
#define D 1024
#define VEC_LEN 16

constexpr int D_head = 1024;
constexpr int input_size = N * (D / VEC_LEN);
constexpr int output_size = N * D_head / 32;

typedef ap_int<16> type_t;
using vec_t = tapa::vec_t<type_t, VEC_LEN>;

void selfAttention(
    tapa::mmap<vec_t> input,
    tapa::mmap<vec_t> WQ,
    tapa::mmap<vec_t> WK,
    tapa::mmap<vec_t> WV,
    tapa::mmap<vec_t> offchip_Q,
    tapa::mmap<vec_t> offchip_K,
    tapa::mmap<vec_t> offchip_V,
    tapa::mmap<vec_t> offchip_scores,
    tapa::mmap<vec_t> offchip_sm_scores,
    tapa::mmap<vec_t> output,
    tapa::mmap<int> cycle_count
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

void to_column_major(const type_t in[D][D], vec_t (&out)[D/VEC_LEN][D]){
    for (int i = 0; i < D / VEC_LEN; i++){
        for (int j = 0; j < D; j++){
            for (int k = 0; k < VEC_LEN; k++){
                out[i][j][k] = in[i*VEC_LEN + k][j];
            }
        }
    }
}

void input_to_tiled(const type_t in[N][D], vec_t (&out)[N][D/VEC_LEN]){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < D / VEC_LEN; j++){
            for (int k = 0; k < VEC_LEN; k++){
                out[i][j][k] = in[i][j*VEC_LEN + k];
            }
        }
    }
}

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    srand((unsigned)time(nullptr));

    // Example input matrix (8x8)
    type_t input_primitive[N][D];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            input_primitive[i][j] = ap_int<16>(1);
        }
    }
    vec_t input[N][D/VEC_LEN];
    input_to_tiled(input_primitive, input);

    type_t WQ_primitive[D][D];
    type_t WK_primitive[D][D];
    type_t WV_primitive[D][D];
    for (int i = 0; i < D; i++){
        for (int j = 0; j < D; j++){
            WQ_primitive[i][j] = ap_int<16>(1);
            WK_primitive[i][j] = ap_int<16>(1);
            WV_primitive[i][j] = ap_int<16>(1);
        }
    }

    vec_t WQ[D/VEC_LEN][D];
    vec_t WK[D/VEC_LEN][D];
    vec_t WV[D/VEC_LEN][D];

    to_column_major(WQ_primitive, WQ);
    to_column_major(WK_primitive, WK);
    to_column_major(WV_primitive, WV);

    aligned_vector<int> cycle_count(1);

    aligned_vector<ap_int<16>> offchip_Q(N * D);
    aligned_vector<ap_int<16>> offchip_K(N * D);
    aligned_vector<ap_int<16>> offchip_V(N * D);
    aligned_vector<ap_int<16>> offchip_scores(N * N);
    aligned_vector<ap_int<16>> offchip_sm_scores(N * N);
    aligned_vector<ap_int<16>> output(N * D);
    
    int64_t kernel_time_ns = tapa::invoke(selfAttention, FLAGS_bitstream,
        tapa::read_only_mmap<ap_int<16>>(input),
        tapa::read_only_mmap<ap_int<16>>(WQ),
        tapa::read_only_mmap<ap_int<16>>(WK),
        tapa::read_only_mmap<ap_int<16>>(WV),

        tapa::read_write_mmap<ap_int<16>>(offchip_Q).reinterpret<vec_t>(),
        tapa::read_write_mmap<ap_int<16>>(offchip_K).reinterpret<vec_t>(),
        tapa::read_write_mmap<ap_int<16>>(offchip_V).reinterpret<vec_t>(),
        tapa::read_write_mmap<ap_int<16>>(offchip_scores).reinterpret<vec_t>(),
        tapa::read_write_mmap<ap_int<16>>(offchip_sm_scores).reinterpret<vec_t>(),
        
        tapa::write_only_mmap<ap_int<16>>(output).reinterpret<vec_t>(),
        tapa::write_only_mmap<int>(cycle_count)
    );

    std::cout << "Cycle count: " << cycle_count[0] << std::endl;
    std::cout << "Kernel time (ns): " << kernel_time_ns << std::endl;
    std::cout << "Kernel time (us): " << float(kernel_time_ns)/1000.0 << std::endl;
    std::cout << "Kernel time (ms): " << float(kernel_time_ns)/1000000.0 << std::endl;

    // print out the offichip_Q
    for (int i = 0; i < N; i++){
        for (int j = 0; j < D; j++){
            std::cout << offchip_Q[i][j] << '\t';
        }
        std::cout << '\n';
    }

    return 0;
}