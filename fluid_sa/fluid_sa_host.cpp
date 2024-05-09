#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>
#include <tapa.h>
#include <gflags/gflags.h>
#include <ap_int.h>

void fluid_spatial_kernel(
    const int N,
    const int M,
    const int vec1_size,
    const int vec2_size,
    const int mtx1_size,
    const int mtx2_size,
    const int output_size,
    tapa::mmap<int> vec1,
    tapa::mmap<int> vec2,
    tapa::mmap<int> mtx1,
    tapa::mmap<int> mtx2,
    tapa::mmap<int> output_mtx,
    tapa::mmap<int> cycle_count
);

using std::vector;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    srand((unsigned)time(nullptr));

    int N = 256;
    int M = 128;

    // data preparation
    aligned_vector<int> vec1(N*2); // 2 1xN vector Q and K
    aligned_vector<int> mtx1(N*M*3); // half of Wq and full Wk
    aligned_vector<int> vec2(N); // 1xN vector Q
    aligned_vector<int> mtx2(N*M); // half of Wq
    aligned_vector<int> output(M*M*4);
    aligned_vector<int> cycle_count(1);

    vector<int> x_vec(N);
    vector<int> y_vec(N);
    vector<int> Wq(N*M*2);
    vector<int> Wk(N*M*2);
    vector<int> output_golden(M*M*4);

    for(int i = 0; i < N; i++){
        int val = (rand() % 6) + 1;
        vec1[i] = val;
        vec2[i] = val;
        x_vec[i] = val;
    }

    for(int i = 0; i < N; i++){
        int val = (rand() % 6) + 1;
        vec1[N+i] = val;
        y_vec[i] = val;
    }

    for(int i = 0; i < N*M; i++){
        int val = (rand() % 6) + 1;
        mtx2[i] = val;
        Wq[i] = val;
    }

    for(int i = 0; i < N*M*3; i++){
        int val = (rand() % 6) + 1;
        mtx1[i] = val;
        if(i < N*M){
            Wq[i+N*M] = val;
        } else {
            Wk[i-N*M] = val;
        }
    }

    // cpu 
    vector<int> query(M*2);
    vector<int> key(M*2);
    for(int i = 0; i < M*2; i++){
        int q = 0;
        int k = 0;
        for(int j = 0; j < N; j++){
            q += x_vec[j] * Wq[i*N + j];
            k += y_vec[j] * Wk[i*N + j];
        }
        query[i] = q;
        key[i] = k;
    }

    for(int i = 0; i < M*2; i++){
        for(int j = 0; j < M*2; j++){
            output_golden[i*M*2+j] = query[j]*key[i];
        }
    }



    // invoke the kernel

    int64_t kernel_time_ns = tapa::invoke(fluid_spatial_kernel, FLAGS_bitstream,
        N, M, N*2, N, N*M*3, N*M, M*M*4,
        tapa::read_only_mmap<int>(vec1),
        tapa::read_only_mmap<int>(vec2),
        tapa::read_only_mmap<int>(mtx1),
        tapa::read_only_mmap<int>(mtx2),
        tapa::write_only_mmap<int>(output),
        tapa::write_only_mmap<int>(cycle_count));
    
    std::clog << "cycle time: " << cycle_count[0] << std::endl;
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;

    int error = 0;

    // compare
    for(int i = 0; i < M*M*4; i++){
        if(output[i] != output_golden[i]){
            std::clog << "index: " << i << ", actual: " << output[i] << ", expect: " << output_golden[i] << std::endl;
            error++;
        }
    }

    if (error == 0) {
        std::clog << "PASSED" << std::endl;
    } else {
        std::clog << "FAILED" << std::endl;
        return 1;
    }
    
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
    return 0;
        
}

