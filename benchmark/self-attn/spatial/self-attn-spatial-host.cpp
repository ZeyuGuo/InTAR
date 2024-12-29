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

#define VEC_LEN 16
constexpr int N = 256;
constexpr int D = 1024;
constexpr int D_head = 128;

constexpr int D_head_div_2 = D_head / 2;
constexpr int weight_size_cc0 = D * D_head / 16;
constexpr int weight_size_cc1 = D * D_head / 8;
constexpr int input_size = N * D / 16;
constexpr int output_size = N * D_head / 32;

using type_t = ap_int<16>;
using vec_t = tapa::vec_t<type_t, VEC_LEN>;

void selfAttention(
    tapa::mmap<vec_t> input,
    tapa::mmap<vec_t> WQ,
    tapa::mmap<vec_t> WK,
    tapa::mmap<vec_t> WV,
    tapa::mmap<vec_t> output,
    tapa::mmap<int> cycle_count
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int L = argc > 1 ? atoll(argv[1]) : N;

    srand((unsigned)time(nullptr));

    // Example input and weight matrices
    
    aligned_vector<type_t> input(N * D);
    aligned_vector<type_t> WQ(D * D_head);
    aligned_vector<type_t> WK(D * D_head);
    aligned_vector<type_t> WV(D * D_head);
    aligned_vector<type_t> output(N * D_head);

    aligned_vector<int> cycle_count(1);

    int64_t kernel_time_ns = tapa::invoke(selfAttention, FLAGS_bitstream,
        tapa::read_only_mmap<type_t>(input).reinterpret<vec_t>(),
        tapa::read_only_mmap<type_t>(WQ).reinterpret<vec_t>(),
        tapa::read_only_mmap<type_t>(WK).reinterpret<vec_t>(),
        tapa::read_only_mmap<type_t>(WV).reinterpret<vec_t>(),
        tapa::write_only_mmap<type_t>(output).reinterpret<vec_t>(),
        tapa::write_only_mmap<int>(cycle_count));

    std::cout << "Output:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            std::cout << output[i * D + j] << " ";
        }
        std::cout << std::endl; 
    }

    return 0;
}