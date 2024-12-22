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

constexpr int seq_len = 256;
constexpr int D = 1024;
constexpr int D_head = 1024;
constexpr int D_head_div_2 = D_head / 2;
constexpr int weight_size_cc0 = D * D_head / 16;
constexpr int weight_size_cc1 = D * D_head / 8;
constexpr int input_size = seq_len * D / 16;
constexpr int output_size = seq_len * D_head / 32;

using type_t = ap_int<16>;
using int16_vD = tapa::vec_t<type_t, D>;
using int16_vL = tapa::vec_t<type_t, seq_len>;

void selfAttention(
    tapa::mmap<int16_vL> input,
    tapa::mmap<int16_vD> WQ,
    tapa::mmap<int16_vD> WK,
    tapa::mmap<int16_vD> WV,
    tapa::mmap<int16_vL> output
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int L = argc > 1 ? atoll(argv[1]) : seq_len;

    srand((unsigned)time(nullptr));

    // Example input and weight matrices
    
    aligned_vector<ap_int<16>> input(D * L);
    aligned_vector<ap_int<16>> WQ(D * D_head);
    aligned_vector<ap_int<16>> WK(D * D_head);
    aligned_vector<ap_int<16>> WV(D * D_head);
    aligned_vector<ap_int<16>> output(D * L);

    int64_t kernel_time_ns = tapa::invoke(selfAttention, FLAGS_bitstream,
        tapa::read_only_mmap<ap_int<16>>(input).reinterpret<int16_vL>(),
        tapa::read_only_mmap<ap_int<16>>(WQ).reinterpret<int16_vD>(),
        tapa::read_only_mmap<ap_int<16>>(WK).reinterpret<int16_vD>(),
        tapa::read_only_mmap<ap_int<16>>(WV).reinterpret<int16_vD>(),
        tapa::write_only_mmap<ap_int<16>>(output).reinterpret<int16_vD>());

    return 0;
}