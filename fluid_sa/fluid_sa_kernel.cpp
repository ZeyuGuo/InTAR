#include <cmath>
#include <string>
#include <tapa.h>
#include <ap_int.h>

constexpr int D = 256;

void read_mtx(
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

// module-level sharing between stage
void pe_gemv(
    const int N,
    tapa::istream<int>& m_size,
    tapa::istream<int>& fifo_in_a,
    tapa::istream<int>& fifo_in_b,
    tapa::ostream<int>& fifo_out
){
    for(;;){

        int cache[D];
        int M = m_size.read();

        for(int i = 0; i < N;){
            if(!fifo_in_b.empty()){
                int c; fifo_in_b.try_read(c);
                cache[i] = c;
                i++;
            }
        }
        for(int i = 0; i < M; i++){
            int ans = 0;
            for(int j = 0; j < N;){
                if(!fifo_in_a.empty()){
                    int a; fifo_in_a.try_read(a);
                    ans += a * cache[j];
                    j++;
                }
            }
            fifo_out.write(ans);
        }
    }
}

void pe_outer_prod_span(
    const int M,
    tapa::istream<int>& fifo_in_a,
    tapa::istream<int>& fifo_in_b,
    tapa::ostream<int>& fifo_out
){
    for(;;){

        int cache_a[D];
        int cache_b[D];

        for(int i = 0; i < M;){
            // fetch data
            if(!fifo_in_a.empty() & !fifo_in_b.empty()){
                int a; fifo_in_a.try_read(a);
                int b; fifo_in_b.try_read(b);
                for(int j = 0; j < i; j++){
                    int res1 = cache_a[j]*b;
                    int res2 = cache_b[j]*a;
                    fifo_out.write(res1);
                    fifo_out.write(res2);
                }
                fifo_out.write(a*b);
                i++;
            }
        }
    }
}

void pe_outer_prod_cache(
    const int M,
    tapa::istream<int>& fifo_in_a,
    tapa::istream<int>& fifo_in_b,
    tapa::ostream<int>& fifo_out
){
    for(;;){

        int cache_a[D];
        
        // load to cache
        for(int i = 0; i < M;){
            if (!fifo_in_a.empty()){
                int a; fifo_in_a.try_read(a);
                cache_a[i] = a;
                i++;
            }
        }

        // compute
        for(int i = 0; i < M;){
            if (!fifo_in_b.empty()){
                int b; fifo_in_b.try_read(b);
                for(int j = 0; j < M; j++){
                    int res = b * cache_a[j];
                    fifo_out.write(res);
                }
                i++;
            }
        }
    }
}

// loop-level sharing, mutate between stage
void pe_fluid(
    const int N,
    const int M,
    tapa::istream<int>& fifo_in_a,
    tapa::istream<int>& fifo_in_b,
    tapa::istream<int>& fifo_in_pe,
    tapa::ostream<int>& fifo_out,
    // interface to dsp
    tapa::ostream<ap_uint<64>>& fifo_dsp_ops,
    tapa::istream<int>& fifo_ret_val
){
    for(;;){

        int cache_stage1[D];
        int cache_stage2[D];

        #pragma HLS array_partition variable=cache_stage2 cyclic factor=4

        // stage 1: GEMV
load_vec:
        for(int i = 0; i < N;){
            if(!fifo_in_b.empty()){
                int c; fifo_in_b.try_read(c);
                cache_stage1[i] = c;
                i++;
            }
        }

        for(int i = 0; i < M; i++){
            int ans = 0;

gemv:
            for(int i_req = 0, i_resp = 0; i_resp < N;){
                if((i_req < N) & !fifo_in_a.empty()){
                    int a; fifo_in_a.try_read(a);
                    ap_uint<64> ops = 0;
                    ops(63, 32) = tapa::bit_cast<ap_uint<32>>(a);
                    ops(31, 0) = tapa::bit_cast<ap_uint<32>>(cache_stage1[i_req]);
                    fifo_dsp_ops.write(ops);
                    i_req++;
                }
                if(!fifo_ret_val.empty()){
                    int val; fifo_ret_val.try_read(val);
                    ans += val;
                    i_resp++;
                }
            }
            cache_stage2[i] = ans;
            cache_stage2[M+i] = fifo_in_pe.read();
        }

        // stage 2: outer product
        // compute
        for(int i = 0; i < M*2; i++){
            int b = fifo_in_pe.read();

outer_prod:
            for(int i_req = 0, i_resp = 0; i_resp < M*2;){
                #pragma HLS pipeline II=1

                if(i_req < M*2){
                    ap_uint<64> ops = 0;
                    ops(63, 32) = tapa::bit_cast<ap_uint<32>>(b);
                    ops(31, 0) = tapa::bit_cast<ap_uint<32>>(cache_stage2[i_req]);
                    fifo_dsp_ops.write(ops);
                    i_req++;
                }
                if(!fifo_ret_val.empty()){
                    int res; fifo_ret_val.try_read(res);
                    fifo_out.write(res);
                    i_resp++;
                }
            }
        }
    }
}

void pe_loop_shared(
    tapa::istream<ap_uint<64>>& fifo_in,
    tapa::ostream<int>& fifo_out
){
    for(;;){
        if(!fifo_in.empty() & !fifo_out.full()){
            ap_uint<64> inp; fifo_in.try_read(inp);
            int a = tapa::bit_cast<int>((ap_uint<32>)inp(63, 32));
            int b = tapa::bit_cast<int>((ap_uint<32>)inp(31, 0));
            int res = a * b;
            fifo_out.try_write(res);
        }
    }
}

void send_signal(
    const int M,
    tapa::ostream<int>& fifo_out
){
    fifo_out.write(M);
    fifo_out.write(M*2);
}


void measure_cycle(tapa::istream<bool>& fifo_fin, tapa::mmap<int> cycle_count){
    for(int cycle = 0;;cycle++){
        if(!fifo_fin.empty()){
            fifo_fin.read(nullptr);
            cycle_count[0] = cycle;
            break;
        }
    }
}

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
){
    tapa::stream<int> fifo_vec1("fifo_vec1");
    tapa::stream<int> fifo_vec2("fifo_vec2");
    tapa::stream<int> fifo_mtx1("fifo_mtx1");
    tapa::stream<int> fifo_mtx2("fifo_mtx2");
    tapa::stream<int> fifo_pe_interconnect("fifo_pe_interconnect");
    tapa::stream<ap_uint<64>, 6> fifo_dsp_ops("fifo_dsp_ops");
    tapa::stream<int> fifo_ret_val("fifo_ret_val");
    tapa::stream<int> fifo_output("fifo_output");
    tapa::stream<bool> fifo_fin("fifo_fin");
    tapa::stream<int> fifo_signal("fifo_signal");

    tapa::task()
        .invoke<tapa::join>(read_mtx, vec1_size, vec1, fifo_vec1)
        .invoke<tapa::join>(read_mtx, vec2_size, vec2, fifo_vec2)
        .invoke<tapa::join>(read_mtx, mtx1_size, mtx1, fifo_mtx1)
        .invoke<tapa::join>(read_mtx, mtx2_size, mtx2, fifo_mtx2)
        .invoke<tapa::join>(send_signal, M, fifo_signal)
        .invoke<tapa::detach>(pe_gemv, N, fifo_signal, fifo_mtx1, fifo_vec1, fifo_pe_interconnect)
        .invoke<tapa::detach>(
            pe_fluid, 
            N, M, 
            fifo_mtx2, 
            fifo_vec2, 
            fifo_pe_interconnect, 
            fifo_output, 
            fifo_dsp_ops,
            fifo_ret_val)
        .invoke<tapa::detach>(pe_loop_shared, fifo_dsp_ops, fifo_ret_val)
        .invoke<tapa::join>(write_mtx, output_size, output_mtx, fifo_output, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}