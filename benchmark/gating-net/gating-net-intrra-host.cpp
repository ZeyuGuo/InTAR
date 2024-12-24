#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>
#include <cmath>
#include <tapa.h>
#include <gflags/gflags.h>
#include <ap_int.h>

constexpr int batch_size = 32; // Number of input examples
constexpr int batch_size_div_2 = batch_size / 2;
constexpr int input_dim = 4096;  // Input dimension
constexpr int hidden_dim = 11008; // Hidden dimension
constexpr int weight_size = input_dim * hidden_dim / 8;
constexpr int input_size_cc0 = batch_size * input_dim;
constexpr int input_size_cc1 = batch_size * input_dim / 2;
constexpr int output_size = batch_size * input_dim; 


using int16_v16 = tapa::vec_t<ap_int<16>, 16>;

void gatingNet(
    tapa::mmap<int16_v16> X_acc0,
    tapa::mmap<int16_v16> X_acc1,
    tapa::mmap<int16_v16> W_acc0,
    tapa::mmap<int16_v16> W_acc1,
    tapa::mmap<int16_v16> acc1_out,
    tapa::mmap<int> cycle_count
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int L = argc > 1 ? atoll(argv[1]) : batch_size;

    srand((unsigned)time(nullptr));

    // Example input and weight matrices
    
    aligned_vector<ap_int<16>> input1(input_size_cc0);
    aligned_vector<ap_int<16>> input2(input_size_cc1);
    aligned_vector<ap_int<16>> weight_cc0(weight_size*16);
    aligned_vector<ap_int<16>> weight_cc1(weight_size*16);
    aligned_vector<ap_int<16>> acc1_out(output_size);
    aligned_vector<int> cycle_count(1);

    int64_t kernel_time_ns = tapa::invoke(gatingNet, FLAGS_bitstream,
        tapa::read_only_mmap<ap_int<16>>(input1).reinterpret<int16_v16>(),
        tapa::read_only_mmap<ap_int<16>>(input2).reinterpret<int16_v16>(),
        tapa::read_only_mmap<ap_int<16>>(weight_cc0).reinterpret<int16_v16>(),
        tapa::read_only_mmap<ap_int<16>>(weight_cc1).reinterpret<int16_v16>(),
        tapa::write_only_mmap<ap_int<16>>(acc1_out).reinterpret<int16_v16>(),
        tapa::write_only_mmap<int>(cycle_count));

    std::cout << "Cycle count: " << cycle_count[0] << std::endl;
    std::cout << "Latency: " << kernel_time_ns * 1e-9 << " s" << std::endl;

    return 0;
}