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
    tapa::mmaps<ap_int<27>, 3> w,
    tapa::mmaps<ap_int<18>, 3> a,
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

    const int len = 27;
    const int width = 3;

    // example weight, activation, and scale
    vector<vector<int>> weight(width);
    vector<vector<int>> activation(width);
    aligned_vector<float> scale(len);

    for(int i = 0; i < width; i++){
        for(int j = 0; j < len*3; j++){
            weight[i].push_back(-2);
        }

        for(int j = 0; j < len; j++){
            activation[i].push_back(13);
        }
    }

    for(int i = 0; i < len; i++){
        scale[i] = 3.5;
    }

    // preprocess weight -> ap_int<27>
    
    aligned_vector<aligned_vector<ap_int<27>>> weight_pack(3);
    for(int l = 0; l < width; l++){
        for(int i = 0; i < len; i++){
            int val = weight[l][i*3] + (weight[l][i*3 + 1] << 12) + (weight[l][i*3+2] << 24);
            ap_int<32> full = tapa::bit_cast<ap_int<32>>(val);
            weight_pack[l].push_back((ap_int<27>) full(26, 0));
        }
    }
    // preprocess activation -> ap_int<18>
    aligned_vector<aligned_vector<ap_int<18>>> activation_pack(3);
    for(int l = 0; l < width; l++){
        for(int i = 0; i < len; i++){
            ap_int<32> full = tapa::bit_cast<ap_int<32>>(activation[l][i]);
            activation_pack[l].push_back((ap_int<18>) full(17, 0));
        }
    }

    // invoke the kernel

    aligned_vector<float> output(len);
    aligned_vector<int> cycle_count(1);
    int64_t kernel_time_ns = tapa::invoke(w3a6o48linear, FLAGS_bitstream,
        len, len, len,
        tapa::read_only_mmaps<ap_int<27>, 3>(weight_pack),
        tapa::read_only_mmaps<ap_int<18>, 3>(activation_pack),
        tapa::read_only_mmap<float>(scale),
        tapa::write_only_mmap<float>(output),
        tapa::write_only_mmap<int>(cycle_count));
    
    std::clog << "cycle time: " << cycle_count[0] << std::endl;
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;
    
    // for(int i = 0; i < len; i++){
    //     std::clog << output[i] << std::endl;
    // }

    // // ground truth
    // for (int l = 0; l < 3; l++){
    //     for(int i = 0; i < 9; i++){
    //         float sum = 0;
    //         for(int j = 0; j < 3; j++){
    //             for(int k = 0; k < 3; k++){
    //                 sum+=weight[l][i*9+j*3+k] * activation[l][i*3+j];
    //             }
    //         }
    //         sum*=scale[l*9+i];
    //         std::clog << "element " << i << ": " << sum << std::endl;
    //     }
    // }
        
}

