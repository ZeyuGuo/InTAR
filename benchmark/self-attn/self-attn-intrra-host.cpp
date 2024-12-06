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
constexpr float scale = 1/32.0;

using float_v16 = tapa::vec_t<float, 16>;

void selfAttention(
    tapa::mmap<float_v16> X_acc0,
    tapa::mmap<float_v16> X_acc1,
    tapa::mmap<float_v16> W_acc0,
    tapa::mmap<float_v16> W_acc1,
    tapa::mmap<float_v16> acc0_out,
    tapa::mmap<float_v16> acc1_out
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int L = argc > 1 ? atoll(argv[1]) : seq_len;

    srand((unsigned)time(nullptr));

    // Example input and weight matrices
    
    aligned_vector<float> input1(seq_len * D);
    aligned_vector<float> input2(seq_len * D);
    aligned_vector<float> weight_cc0(weight_size_cc0*16);
    aligned_vector<float> weight_cc1(weight_size_cc1*16);
    aligned_vector<float> acc0_out(seq_len * D_head / 2);
    aligned_vector<float> acc1_out(seq_len * D_head / 2);

    int64_t kernel_time_ns = tapa::invoke(selfAttention, FLAGS_bitstream,
        tapa::read_only_mmap<float>(input1).reinterpret<float_v16>(),
        tapa::read_only_mmap<float>(input2).reinterpret<float_v16>(),
        tapa::read_only_mmap<float>(weight_cc0).reinterpret<float_v16>(),
        tapa::read_only_mmap<float>(weight_cc1).reinterpret<float_v16>(),
        tapa::read_only_mmap<float>(acc0_out).reinterpret<float_v16>(),
        tapa::read_only_mmap<float>(acc1_out).reinterpret<float_v16>());

    return 0;
}