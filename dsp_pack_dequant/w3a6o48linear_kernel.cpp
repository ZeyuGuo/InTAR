#include <cmath>
#include <string>
#include <tapa.h>
#include <ap_int.h>

void read_w(
    const int LW,
    tapa::async_mmap<ap_int<27>>& w,
    tapa::ostream<ap_int<27>>& weight
){
    for(int i_req = 0, i_resp = 0; i_resp < LW;){
        #pragma HLS pipeline II=1
        if((i_req < LW) & !w.read_addr.full()){
            w.read_addr.write(i_req);
            i_req++;
        }
        if(!w.read_data.empty()){
            ap_int<27> tmp_o; w.read_data.try_read(tmp_o);
            weight.write(tmp_o);
            i_resp++;
        }
    }
}

void read_a(
    const int LA,
    tapa::async_mmap<ap_int<18>>& a,
    tapa::ostream<ap_int<18>>& act
){
    for(int i_req = 0, i_resp = 0; i_resp < LA;){
        #pragma HLS pipeline II=1
        if((i_req < LA) & !a.read_addr.full()){
            a.read_addr.write(i_req);
            i_req++;
        }
        if(!a.read_data.empty()){
            ap_int<18> tmp_o; a.read_data.try_read(tmp_o);
            act.write(tmp_o);
            i_resp++;
        }
    }
}


void read_scale(
    const int LS,
    tapa::async_mmap<float>& scale,
    tapa::ostream<float>& fifo_scale
){
    for(int i_req = 0, i_resp = 0; i_resp < LS;){
        #pragma HLS pipeline II=1
        if((i_req < LS) & !scale.read_addr.full()){
            scale.read_addr.write(i_req);
            i_req++;
        }
        if(!scale.read_data.empty()){
            float tmp_o; scale.read_data.try_read(tmp_o);
            fifo_scale.write(tmp_o);
            i_resp++;
        }
    }
}

void compute(
    tapa::istream<ap_int<27>>& weight,
    tapa::istream<ap_int<18>>& act,
    tapa::ostream<ap_int<48>>& output
){
    // read in weight and activation
    ap_int<48> z = 0;
    for(int count = 0;;){
        #pragma HLS pipeline II=1
        if(!weight.empty() & !act.empty() & (count < 3)){
            ap_int<27> w; weight.try_read(w);
            ap_int<18> a; act.try_read(a);
            ap_int<48> postadd;

            #pragma HLS bind_op variable=z op=mul impl=dsp
            postadd = w * a;
            z += postadd;
            count++;
        }
        if((count == 3) & !output.full()){
            output.try_write(z);
            count = 0;
            z = 0;
        }
    }
}

void dequantize(
    tapa::istreams<ap_int<48>, 3>& z,
    tapa::istream<float>& scale,
    tapa::ostream<float>& output
){
    for(int c_idx = 0;;){
        #pragma HLS pipeline II=1
        
        if(!z[c_idx].empty() & !scale.empty()){
            ap_int<48> val; z[c_idx].try_read(val);
            float s; scale.try_read(s);
            // unpack val
            ap_int<12> z0 = val(11, 0);
            if(z0[11] == 1){
                ap_int<48> sign = -1;
                sign(11, 0) = (ap_int<12>) 0;
                val -= sign;
            }
            ap_int<12> z1 = val(23, 12);
            if(z1[11] == 1){
                ap_int<48> sign = -1;
                sign(23, 0) = 0;
                val -= sign;
            }
            ap_int<12> z2 = val(35, 24);

            int sum = z0 + z1 + z2;
            float out = s * (float)sum;
            output.write(out);
            c_idx++;
            if(c_idx == 3) c_idx = 0;
        }
    }
}

void write_o(
    const int LO,
    tapa::async_mmap<float>& o,
    tapa::istream<float>& fifo_output,
    tapa::ostream<bool>& fifo_fin // measure cycle
){
    for(int i_req = 0, i_resp = 0; i_resp < LO;){
        #pragma HLS pipeline II=1
        if((i_req < LO) & !fifo_output.empty() & !o.write_addr.full() & !o.write_data.full()){
            o.write_addr.try_write(i_req);
            float tmp; fifo_output.try_read(tmp);
            o.write_data.try_write(tmp);
            ++i_req;
        }
        if(!o.write_resp.empty()){
            i_resp += unsigned(o.write_resp.read(nullptr))+1;
        }
    }
    fifo_fin.write(true);
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

void w3a6o48linear(
    const int LW,
    const int LA,
    const int LS,
    tapa::mmaps<ap_int<27>, 3> w,
    tapa::mmaps<ap_int<18>, 3> a,
    tapa::mmap<float> scale,
    tapa::mmap<float> o,
    tapa::mmap<int> cycle_count
){
    tapa::streams<ap_int<27>, 3> fifo_weight("fifo_weight");
    tapa::streams<ap_int<18>, 3> fifo_act("fifo_act");
    tapa::stream<float> fifo_scale("fifo_scale");
    tapa::stream<float> fifo_output("fifo_output");
    tapa::streams<ap_int<48>, 3> fifo_deq("fifo_deq");
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join, 3>(read_w, LW, w, fifo_weight)
        .invoke<tapa::join, 3>(read_a, LA, a, fifo_act)
        .invoke<tapa::join>(read_scale, LS, scale, fifo_scale)
        .invoke<tapa::detach, 3>(compute, fifo_weight, fifo_act, fifo_deq)
        .invoke<tapa::detach>(dequantize, fifo_deq, fifo_scale, fifo_output)
        .invoke<tapa::join>(write_o, LS, o, fifo_output, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}