#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <tapa.h>
#include <gflags/gflags.h>
#include <ap_int.h>

void w3a6o48linear(
    const int LW,
    const int LA,
    const int LS,
    tapa::mmap<ap_int<27>> w,
    tapa::mmap<ap_int<18>> a,
    tapa::mmap<float> scale,
    tapa::mmap<float> output,
    tapa::mmap<int> cycle_count
);

using std::vector;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // example weight, activation, and scale
    vector<int> weight = {0, -3, 1, 2, -1, 1, -3, 1, 2, 0, -3, 1, 2, -1, 1, -3, 1, 2, 0, -3, 1, 2, -1, 1, -3, 1, 2,
                            0, -3, 1, 2, -1, 1, -3, 1, 2, 0, -3, 1, 2, -1, 1, -3, 1, 2, 0, -3, 1, 2, -1, 1, -3, 1, 2,};
    vector<int> activation = {8, -15, -51, 8, -15, -51, 8, -15, -51,
                                8, -15, -51, 8, -15, -51, 8, -15, -51,};
    aligned_vector<float> scale = {3.5, 3.5, 3.5, 3.5, 3.5, 3.5};

    // preprocess weight -> ap_int<27>
    
    aligned_vector<ap_int<27>> weight_pack(18);
    for(int i = 0; i < 18; i++){
        int val = weight[i*3] + (weight[i*3 + 1] << 12) + (weight[i*3+2] << 24);
        ap_int<32> full = tapa::bit_cast<ap_int<32>>(val);
        weight_pack[i] = (ap_int<27>) full(26, 0);
    }

    // preprocess activation -> ap_int<18>
    aligned_vector<ap_int<18>> activation_pack(18);
    for(int i = 0; i < 18; i++){
        ap_int<32> full = tapa::bit_cast<ap_int<32>>(activation[i]);
        activation_pack[i] = (ap_int<18>) full(17, 0);
    }

    // invoke the kernel

    aligned_vector<float> output(6);
    aligned_vector<int> cycle_count(1);
    int64_t kernel_time_ns = tapa::invoke(w3a6o48linear, FLAGS_bitstream,
        18, 18, 6,
        tapa::read_only_mmap<ap_int<27>>(weight_pack),
        tapa::read_only_mmap<ap_int<18>>(activation_pack),
        tapa::read_only_mmap<float>(scale),
        tapa::write_only_mmap<float>(output),
        tapa::write_only_mmap<int>(cycle_count));
    
    std::clog << "cycle time: " << cycle_count[0] << std::endl;
    
    for(int i = 0; i < 3; i++){
        std::clog << output[i] << std::endl;
    }

    // ground truth
    for(int i = 0; i < 3; i++){
        float sum = 0;
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 3; k++){
                sum+=weight[i*9+j*3+k] * activation[i*3+j];
            }
        }
        sum*=scale[i];
        std::clog << "element " << i << ": " << sum << std::endl;
    }
        
}

