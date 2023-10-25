#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <iostream>
#include <string>
#include <tapa.h>

using tensor1d = std::vector<float>;
using tensor2d = std::vector<tensor1d>;
using tensor3d = std::vector<tensor2d>;

using float_v2 = tapa::vec_t<float, 2>;

constexpr float EPS = 1e-5;
constexpr int DIM = 4096;
constexpr int MAX_SEQ_LEN = 512;
constexpr int HIDDEN_DIM = 11008;
constexpr int VOCAB_SIZE = 32000;
constexpr int N_LAYERS = 32;
constexpr int N_HEADS = 32;
constexpr int HEAD_SIZE = 128;

constexpr int FIFO_DEPTH = 2;

template <typename data_t>
inline void bh(tapa::istream<data_t> & q) {
#pragma HLS inline
    for (;;) {
#pragma HLS pipeline II=1
        data_t tmp; q.try_read(tmp);
    }
}

void black_hole_float(tapa::istream<float>& fifo_in){
    bh(fifo_in);
}

void black_hole_bool(tapa::istream<bool>& fifo_in){
    bh(fifo_in);
}

void accum(
    tapa::istream<float>& fifo_x, 
    tapa::istream<float>& fifo_y,
    tapa::ostream<float>& fifo_z
    ) {

    for(;;){
        if(!fifo_x.empty() & !fifo_y.empty() & !fifo_z.full()){
            float tmp_x; fifo_x.try_read(tmp_x);
            float tmp_y; fifo_y.try_read(tmp_y);
            float val = tmp_x + tmp_y;
            fifo_z.try_write(val);
        }
    }
}

void duplicate(
    tapa::istream<float>& fifo_in,
    tapa::ostream<float>& fifo_out_a,
    tapa::ostream<float>& fifo_out_b
){
    for(;;){
        if(!fifo_in.empty() & !fifo_out_a.full() & !fifo_out_b.full()){
            float tmp; fifo_in.try_read(tmp);
            fifo_out_a.try_write(tmp);
            fifo_out_b.try_write(tmp);
        }
    }
}

void rmsnorm(
    tapa::ostream<float>& output, 
    tapa::istream<float>& input, 
    tapa::istream<float>& weight
    ) {

    for(;;){
        float input_cache[DIM];

        float denom = 0.0;
        for (int i = 0; i < DIM;){
            #pragma HLS pipeline II=1
            if(!input.empty()){
                float tmp; input.try_read(tmp);
                input_cache[i] = tmp;
                denom += tmp * tmp; // TODO: pipeline
                i++;
            }
        }
            
        denom = denom / DIM + EPS;
        float inv = 1 / sqrt(denom);
        
        for (int i = 0; i < DIM;){
            #pragma HLS pipeline II=1
            if(!weight.empty() & !output.full()){
                float tmp_w; weight.try_read(tmp_w);
                float val = input_cache[i] * inv * tmp_w; // TODO: pipeline ?
                output.try_write(val);
                i++;
            }
        }
    }
}

void softmax(
    tapa::ostream<float>& output, 
    tapa::istream<float>& input, 
    tapa::istream<int>& fifo_inst
    ) {

    for(;;){
        float input_cache[MAX_SEQ_LEN];
        
        const int N = fifo_inst.read();

        float sum = 0;
        for (int i = 0; i < N;) {
        #pragma HLS pipeline II=1
            if(!input.empty()){
                float tmp; input.try_read(tmp);
                tmp = exp(tmp);
                sum += tmp;
                input_cache[i] = tmp;
                i++;
            }
        }

        float inv = 1.0 / sum;
        // normalize
        for (int i = 0; i < N;){
        #pragma HLS pipeline II=1
            if(!output.full()){
                float val = input_cache[i] * inv;
                output.try_write(val);
                i++;
            }
        }
    }
}

// used when length of input is not big
// always do 1xN * NxM
void matmul_kqv(
    tapa::istream<float>& input,
    tapa::ostream<float>& input_out,
    tapa::istream<float>& weight,
    tapa::ostream<float>& output,
    const int N,
    const int M
){
    for(int l = 0; l < N_LAYERS; l++){
        float input_cache[DIM];
        for(int i = 0; i < N;){
            if(!input.empty() & !input_out.full()){
                float tmp; input.try_read(tmp);
                input_out.try_write(tmp);
                input_cache[i] = tmp;
                i++;
            }
        }

        for(int i = 0; i < M; i++){
            float val = 0.0;
            for(int j = 0; j < N;){
                if(!weight.empty()){
                    float tmp_w; weight.try_read(tmp_w);
                    val += tmp_w * input_cache[j];
                    j++;
                }
            }
            output.write(val);
        }
    }
}

void matmul_w2(
    tapa::istream<float>& input,
    tapa::istream<float>& weight,
    tapa::ostream<float>& output
){
    for(int l = 0; l < N_LAYERS; l++){
        float input_cache[HIDDEN_DIM];
        for(int i = 0; i < HIDDEN_DIM;){
            if(!input.empty()){
                float tmp; input.try_read(tmp);
                input_cache[i] = tmp;
                i++;
            }
        }

        for(int i = 0; i < DIM; i++){
            float val = 0.0;
            for(int j = 0; j < HIDDEN_DIM;){
                if(!weight.empty()){
                    float tmp_w; weight.try_read(tmp_w);
                    val += tmp_w * input_cache[j];
                    j++;
                }
            }
            output.write(val);
        }
    }
}

void read_token_embedding(
    const int token_index,
    tapa::async_mmap<float>& token_embedding_table,
    tapa::ostream<float>& output
){
    for(int i_req = token_index * DIM, i_resp = 0; i_resp < DIM;){
        #pragma HLS pipeline II=1
        if((i_req < (token_index + 1) * DIM) & !token_embedding_table.read_addr.full()){
            token_embedding_table.read_addr.write(i_req);
            i_req++;
        }
        if(!token_embedding_table.read_data.empty()){
            float tmp_o; token_embedding_table.read_data.try_read(tmp_o);
            output.write(tmp_o);
            i_resp++;
        }
    }
}

// rotary positional embedding for query and key
void RoPE(
    tapa::istream<float_v2>& query_in,
    tapa::istream<float_v2>& key_in,
    tapa::istream<float_v2>& freq_cis,
    tapa::ostream<float_v2>& query_out,
    tapa::ostream<float_v2>& key_out
){
    for(;;){
        if(!freq_cis.empty() & !query_in.empty() & !key_in.empty() & !query_out.full() & !key_out.full()){
            float_v2 q; query_in.try_read(q);
            float_v2 k; key_in.try_read(k);
            float_v2 freq; freq_cis.try_read(freq);
            float_v2 qo; float_v2 ko;
            qo[0] = q[0] * freq[0] - q[1] * freq[1];
            qo[1] = q[0] * freq[1] + q[1] * freq[0];
            ko[0] = k[0] * freq[0] - k[1] * freq[1];
            ko[1] = k[0] * freq[1] + k[1] * freq[0];
            query_out.try_write(qo);
            key_out.try_write(ko);
        }
    }
}

void silu(
    tapa::istream<float>& input,
    tapa::ostream<float>& output
) {
    for(;;){
        if(!input.empty() & !output.full()){
            float val; input.try_read(val);
            val = val / (1.0 + std::exp(-val));
            output.try_write(val);
        }
    }
}

void ele_mul(
    tapa::istream<float>& fifo_x,
    tapa::istream<float>& fifo_y,
    tapa::ostream<float>& output
) {
    for(;;){
        if(!fifo_x.empty() & !fifo_y.empty() & !output.full()){
            float x; fifo_x.try_read(x);
            float y; fifo_y.try_read(y);
            float val = x * y;
            output.try_write(val);
        }
    }
}

void kv_cache(
    const int token_position,
    tapa::istream<float>& input,
    tapa::async_mmap<float>& output,
    tapa::ostream<bool> done
) {
    for(int l = 0; l < N_LAYERS; l++){
        int offset = l*MAX_SEQ_LEN*DIM + token_position*DIM;
        for(int i_req = offset, i_resp = 0; i_resp < DIM;){
            #pragma HLS pipeline II=1
            if((i_req < offset+DIM) & !input.empty() & !output.write_addr.full() & !output.write_data.full()){
                output.write_addr.try_write(i_req);
                float tmp; input.try_read(tmp);
                output.write_data.try_write(tmp);
                ++i_req;
            }
            if(!output.write_resp.empty()){
                i_resp += unsigned(output.write_resp.read(nullptr))+1;
            }
        }

        done.write(true);
    }
}

void read_cache(
    const int token_position,
    tapa::istream<bool>& done,
    tapa::async_mmap<float>& keys,
    tapa::ostream<float>& fifo_key
){
    for(int l = 0; l < N_LAYERS; l++){
        done.read();
        for(int i = 0; i < N_HEADS; i++){
            for(int j = 0; j <= token_position; j++){
                int offset = l*DIM*MAX_SEQ_LEN+j*DIM+i*HEAD_SIZE;
                for(int i_req = offset, i_resp = 0; i_resp < HEAD_SIZE;){
                    #pragma HLS pipeline II=1
                    if((i_req < offset+HEAD_SIZE) & !keys.read_addr.full()){
                        keys.read_addr.write(i_req);
                        i_req++;
                    }
                    if(!keys.read_data.empty()){
                        float tmp_o; keys.read_data.try_read(tmp_o);
                        fifo_key.write(tmp_o);
                        i_resp++;
                    }
                }
            }
        }
    }
}

void matmul_attention(
    const int token_position,
    tapa::istream<float>& query,
    tapa::istream<float>& key,
    tapa::ostream<float>& attention,
    tapa::ostream<int>& fifo_inst
){
    for(int l = 0; l < N_LAYERS; l++){
        // cache query
        float query_cache[DIM];
        for(int i = 0; i < DIM;){
            if(!query.empty()){
                query.try_read(query_cache[i]);
                i++;
            }
        }
        for(int i = 0; i < N_HEADS; i++){
            fifo_inst.write(token_position+1);
            for(int j = 0; j <= token_position; j++){
                float score = 0;
                for (int k = 0; k < HEAD_SIZE;){
                    if(!key.empty()){
                        float tmp; key.try_read(tmp);
                        score += query_cache[i * HEAD_SIZE + k] * tmp;
                        k++;
                    }
                }
                score /= std::sqrt(HEAD_SIZE * 1.0);
                attention.write(score);
            }
        }
    }
}

void matmul_value(
    const int token_position,
    tapa::istream<float>& attention,
    tapa::istream<float>& value,
    tapa::ostream<float>& output
) {
    for(int l = 0; l < N_LAYERS; l++){
        float output_cache[DIM];
        for(int i = 0; i < DIM; i++){
            output_cache[i] = 0;
        }
        for(int i = 0; i < N_HEADS; i++){
            for(int j = 0; j <= token_position; j++){
                float attn = attention.read();
                for(int k = 0; k < HEAD_SIZE;){
                    if(!value.empty()){
                        float tmp_v; value.try_read(tmp_v);
                        output_cache[i * HEAD_SIZE + k] += tmp_v * attn;
                        k++;
                    }
                }
            }

        }
        for(int i = 0; i < DIM;){
            if(!output.full()){
                output.try_write(output_cache[i]);
                i++;
            }
        }
    }
}

void matmul_output(
    tapa::istream<float>& input,
    tapa::istream<float>& weight,
    tapa::ostream<float>& output
){
    for(int l = 0; l < N_LAYERS; l++){
        float input_cache[DIM];
        for(int i = 0; i < DIM;){
            if(!input.empty()){
                float tmp; input.try_read(tmp);
                input_cache[i] = tmp;
                i++;
            }
        }

        for(int i = 0; i < DIM; i++){
            float val = 0.0;
            for(int j = 0; j < DIM;){
                if(!weight.empty()){
                    float tmp_w; weight.try_read(tmp_w);
                    val += tmp_w * input_cache[j];
                    j++;
                }
            }
            output.write(val);
        }
    }
}

void matmul_logits(
    tapa::istream<float>& input,
    tapa::istream<float>& weight,
    tapa::ostream<float>& output
){
    for(int l = 0; l < N_LAYERS; l++){
        float input_cache[DIM];
        for(int i = 0; i < DIM;){
            if(!input.empty()){
                float tmp; input.try_read(tmp);
                input_cache[i] = tmp;
                i++;
            }
        }

        for(int i = 0; i < VOCAB_SIZE; i++){
            float val = 0.0;
            for(int j = 0; j < DIM;){
                if(!weight.empty()){
                    float tmp_w; weight.try_read(tmp_w);
                    val += tmp_w * input_cache[j];
                    j++;
                }
            }
            output.write(val);
        }
    }
}

void layer_control_adapter(
    tapa::istream<float>& fifo_token_emb,
    tapa::istream<float>& fifo_feedback,
    tapa::ostream<float>& fifo_next_layer,
    tapa::ostream<float>& fifo_exit
) {
    for(int i = 0; i < DIM;){
        if(!fifo_token_emb.empty() & !fifo_next_layer.full()){
            float tmp; fifo_token_emb.try_read(tmp);
            fifo_next_layer.try_write(tmp);
            i++;
        }
    }

    // iterate 32 layers
    for(int i = 0; i < (N_LAYERS-1) * DIM;){
        if(!fifo_feedback.empty() & !fifo_next_layer.full()){
            float tmp; fifo_feedback.try_read(tmp);
            fifo_next_layer.try_write(tmp);
            if(i % DIM == 0) LOG(INFO) << i / DIM;
            i++;
        }
    }

    // exit the transformer
    for(int i = 0; i < DIM;){
        if(!fifo_feedback.empty() & !fifo_exit.full()){
            float tmp; fifo_feedback.try_read(tmp);
            fifo_exit.try_write(tmp);
            i++;
        }
    }
}

void read_weight(
    tapa::async_mmap<float>& weight,
    tapa::ostream<float>& fifo_w_out
) {
    for(int i_req = 0, i_resp = 0; i_resp < DIM * N_LAYERS;){
        #pragma HLS pipeline II=1
        if((i_req < DIM * N_LAYERS) & !weight.read_addr.full()){
            weight.read_addr.write(i_req);
            i_req++;
        }
        if(!weight.read_data.empty()){
            float tmp_o; weight.read_data.try_read(tmp_o);
            fifo_w_out.write(tmp_o);
            i_resp++;
        }
    }
}

void read_final_rms_weight(
    tapa::async_mmap<float>& weight,
    tapa::ostream<float>& fifo_w_out
) {
    for(int i_req = 0, i_resp = 0; i_resp < DIM;){
        #pragma HLS pipeline II=1
        if((i_req < DIM) & !weight.read_addr.full()){
            weight.read_addr.write(i_req);
            i_req++;
        }
        if(!weight.read_data.empty()){
            float tmp_o; weight.read_data.try_read(tmp_o);
            fifo_w_out.write(tmp_o);
            i_resp++;
        }
    }
}

void read_final_output_weight(
    tapa::async_mmap<float>& weight,
    tapa::ostream<float>& fifo_w_out
) {
    for(int i_req = 0, i_resp = 0; i_resp < DIM*VOCAB_SIZE;){
        #pragma HLS pipeline II=1
        if((i_req < DIM*VOCAB_SIZE) & !weight.read_addr.full()){
            weight.read_addr.write(i_req);
            i_req++;
        }
        if(!weight.read_data.empty()){
            float tmp_o; weight.read_data.try_read(tmp_o);
            fifo_w_out.write(tmp_o);
            i_resp++;
        }
    }
}

void read_ffn_weight(
    tapa::async_mmap<float>& weight,
    tapa::ostream<float>& fifo_w_out
) {
    for(int i_req = 0, i_resp = 0; i_resp < DIM * N_LAYERS * HIDDEN_DIM;){
        #pragma HLS pipeline II=1
        if((i_req < DIM * N_LAYERS * HIDDEN_DIM) & !weight.read_addr.full()){
            weight.read_addr.write(i_req);
            i_req++;
        }
        if(!weight.read_data.empty()){
            float tmp_o; weight.read_data.try_read(tmp_o);
            fifo_w_out.write(tmp_o);
            i_resp++;
        }
    }
}

void read_freq_cis(
    const int token_position,
    tapa::async_mmap<float_v2>& freq_cis,
    tapa::ostream<float_v2>& fifo_freq_cis
){
    for(int l = 0; l < N_LAYERS; l++){
        for(int i_req = token_position*HEAD_SIZE/2, i_resp = 0; i_resp < HEAD_SIZE/2;){
            #pragma HLS pipeline II=1
            if((i_req < (token_position+1)*HEAD_SIZE/2) & !freq_cis.read_addr.full()){
                freq_cis.read_addr.write(i_req);
                i_req++;
            }
            if(!freq_cis.read_data.empty()){
                float_v2 tmp_o; freq_cis.read_data.try_read(tmp_o);
                fifo_freq_cis.write(tmp_o);
                i_resp++;
            }
        }
    }
}

void repeat_freq_cis(
    tapa::istream<float_v2>& fifo_freq_cis_in,
    tapa::ostream<float_v2>& fifo_freq_cis_out
){
    for(;;){
        float_v2 cache[HEAD_SIZE/2];
        for(int i = 0; i < HEAD_SIZE/2; ){
            if(!fifo_freq_cis_in.empty()){
                fifo_freq_cis_in.try_read(cache[i]);
                i++;
            }
        }

        for(int i = 0; i < N_HEADS; i++){
            for(int j = 0; j < HEAD_SIZE/2;){
                if(!fifo_freq_cis_out.full()){
                    fifo_freq_cis_out.try_write(cache[j]);
                    j++;
                }
            }
        }
    }
}

void read_weight_kqv(
    tapa::async_mmap<float>& weight,
    tapa::ostream<float>& fifo_w_out
) {
    for(int i_req = 0, i_resp = 0; i_resp < DIM * DIM * N_LAYERS;){
        #pragma HLS pipeline II=1
        if((i_req < DIM * DIM * N_LAYERS) & !weight.read_addr.full()){
            weight.read_addr.write(i_req);
            i_req++;
        }
        if(!weight.read_data.empty()){
            float tmp_o; weight.read_data.try_read(tmp_o);
            fifo_w_out.write(tmp_o);
            i_resp++;
        }
    }
}

void write_logits(
    tapa::istream<float>& fifo_logits,
    tapa::async_mmap<float>& output
) {
    for(int i_req = 0, i_resp = 0; i_resp < VOCAB_SIZE;){
        #pragma HLS pipeline II=1
        if((i_req < VOCAB_SIZE) & !fifo_logits.empty() & !output.write_addr.full() & !output.write_data.full()){
            output.write_addr.try_write(i_req);
            float tmp; fifo_logits.try_read(tmp);
            output.write_data.try_write(tmp);
            LOG(INFO) << tmp;
            ++i_req;
        }
        if(!output.write_resp.empty()){
            i_resp += unsigned(output.write_resp.read(nullptr))+1;
        }
    }
}

// TODO: merge it into other modules
void float_to_float_vec(
    tapa::istream<float>& fifo_in,
    tapa::ostream<float_v2>& fifo_out
){
    float_v2 tmp;
    for(int c_idx = 0;;){
        if(!fifo_in.empty()){
            fifo_in.try_read(tmp[c_idx]);
            c_idx++;
        }
        if(c_idx == 2) {
            fifo_out.write(tmp);
            c_idx = 0;
        }
    }
}

void float_vec_to_float(
    tapa::istream<float_v2>& fifo_in,
    tapa::ostream<float>& fifo_out
){
    for(;;){
        if(!fifo_in.empty()){
            float_v2 tmp; fifo_in.try_read(tmp);
            for(int i = 0; i < 2; i++){
                fifo_out.write(tmp[i]);
            }
        }
    }
}

/*
void transformer(int token_index, int token_position, Config &config, RunState &state, TransformerWeights &transformer_weights) {
    // a few convenience variables
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int head_size = dim / config.n_heads;

    // copy the token embedding into x
    copy(state.x, transformer_weights.token_embedding_table[token_index]);

    for (int layer = 0; layer < config.n_layers; ++layer) {
        // attention rmsnorm
        rmsnorm(state.xb, state.x, transformer_weights.rms_att_weight[layer]);

        // attention
        matmul(state.q, state.xb, transformer_weights.wq[layer]);
        matmul(state.k, state.xb, transformer_weights.wk[layer]);
        matmul(state.v, state.xb, transformer_weights.wv[layer]);
        
        // apply RoPE positional embeddings
        for (int head = 0; head < config.n_heads; ++head) {
            int start = head * head_size;
            for (int i = 0; i < head_size; i += 2) {
                float q0 = state.q[start + i];
                float q1 = state.q[start + i + 1];
                float k0 = state.k[start + i];
                float k1 = state.k[start + i + 1];
                float fcr = transformer_weights.freq_cis_real[token_position][i / 2];
                float fci = transformer_weights.freq_cis_imag[token_position][i / 2];
                state.q[start + i]     = q0 * fcr - q1 * fci;
                state.q[start + i + 1] = q0 * fci + q1 * fcr;
                state.k[start + i]     = k0 * fcr - k1 * fci;
                state.k[start + i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key/value in cache
        copy(state.key_cache[layer][token_position], state.k);
        copy(state.value_cache[layer][token_position], state.v);

        // multiquery attention
        for (int head = 0; head < config.n_heads; ++head) {
            for (int timestep = 0; timestep <= token_position; ++timestep) {
                float score = 0;
                for (int i = 0; i < head_size; ++i)
                    score += state.q[head * head_size + i] * state.key_cache[layer][timestep][head * head_size + i];
                score /= std::sqrt(head_size * 1.0);
                state.attention[timestep] = score;
            }

            // softmax
            softmax(state.attention, state.attention, token_position+1);

            // weighted sum
            for (int i = 0; i < head_size; ++i) {
                state.xb[head * head_size + i] = 0;
                for (int timestep = 0; timestep <= token_position; ++timestep)
                    state.xb[head * head_size + i] += state.attention[timestep] * state.value_cache[layer][timestep][head * head_size + i];
            }
        }

        // final matmul to get the output of the attention
        matmul(state.xb2, state.xb, transformer_weights.wo[layer]);

        // residual connection back into x
        accum(state.x, state.xb2);

        // ffn rmsnorm
        rmsnorm(state.xb, state.x, transformer_weights.rms_ffn_weight[layer]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x))) * self.w3(x)
        // first calculate self.w1(x) and self.w3(x)
        matmul(state.hb, state.xb, transformer_weights.w1[layer]);
        matmul(state.hb2, state.xb, transformer_weights.w3[layer]);

        // F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; ++i)
            state.hb[i] = state.hb[i] * (1.0 / (1.0 + std::exp(-state.hb[i])));

        // elementwise multiple with w3(x)
        for (int i = 0; i < hidden_dim; ++i)
            state.hb[i] = state.hb[i] * state.hb2[i];
        
        // final matmul to get the output of the ffn
        matmul(state.xb, state.hb, transformer_weights.w2[layer]);

        // residual connection
        accum(state.x, state.xb);
    }

    // final rmsnorm
    rmsnorm(state.x, state.x, transformer_weights.rms_final_weight);

    // classifier into logits
    matmul(state.logits, state.x, transformer_weights.output_weight);
}
*/

void prober(
    tapa::istream<float>& fifo_in
){
    for(;;){
        if(!fifo_in.empty()){
            float tmp; fifo_in.try_read(tmp);
            LOG(INFO) << tmp;
        }
    }
}

// top kernel
void Llama2(
    const int token_index,
    const int token_position,
    tapa::mmap<float> token_embedding_table,
    // TODO: merge these weights as much as possible
    tapa::mmap<float> rms_att_weight,
    tapa::mmap<float> rms_ffn_weight,
    tapa::mmap<float> rms_final_weight,
    tapa::mmap<float> wq,
    tapa::mmap<float> wk,
    tapa::mmap<float> wv,
    tapa::mmap<float> wo,
    tapa::mmap<float> w1,
    tapa::mmap<float> w2,
    tapa::mmap<float> w3,
    tapa::mmap<float_v2> freq_cis,
    tapa::mmap<float> key_cache,
    tapa::mmap<float> value_cache,
    tapa::mmap<float> output_weight,
    tapa::mmap<float> output
){

    tapa::stream<float, FIFO_DEPTH> token_emb("token_emb");
    tapa::stream<float, 2048> fifo_layer_in("fifo_layer_in");
    tapa::stream<float, 2048> fifo_layer_out("fifo_layer_out");
    tapa::stream<float, FIFO_DEPTH> fifo_exit("fifo_exit");
    tapa::stream<float, FIFO_DEPTH> rms_att_w("rms_att_w");
    tapa::stream<float, FIFO_DEPTH> rms_ffn_w("rms_ffn_w");
    tapa::stream<float, FIFO_DEPTH> rms_final_w("rms_final_w");
    tapa::streams<float, 4, FIFO_DEPTH> fifo_rms_to_kqv("fifo_rms_to_kqv"); 
    tapa::streams<float, 4, FIFO_DEPTH> fifo_rms_to_ffn("fifo_rms_to_ffn");
    tapa::stream<float, FIFO_DEPTH> fifo_rms_to_output("fifo_rms_to_output");
    tapa::stream<float, FIFO_DEPTH> fifo_wq("fifo_wq");
    tapa::stream<float, FIFO_DEPTH> fifo_wk("fifo_wk");
    tapa::stream<float, FIFO_DEPTH> fifo_wv("fifo_wv");
    tapa::stream<float, FIFO_DEPTH> fifo_wo("fifo_wo");
    tapa::stream<float, FIFO_DEPTH> fifo_w1("fifo_w1");
    tapa::stream<float, FIFO_DEPTH> fifo_w2("fifo_w2");
    tapa::stream<float, FIFO_DEPTH> fifo_w3("fifo_w3");
    tapa::stream<float, FIFO_DEPTH> fifo_output_w("fifo_output_w");
    
    tapa::streams<float, 3, FIFO_DEPTH> fifo_query("fifo_query");
    tapa::streams<float, 3, FIFO_DEPTH> fifo_key("fifo_key");
    tapa::streams<float_v2, 2, FIFO_DEPTH> fifo_query_v2("fifo_query_v2");
    tapa::streams<float_v2, 2, FIFO_DEPTH> fifo_key_v2("fifo_key_v2");
    tapa::streams<float_v2, 2, FIFO_DEPTH> fifo_freq_cis("fifo_freq_cis");
    tapa::streams<float, 2, FIFO_DEPTH> fifo_value("fifo_value");
    tapa::stream<bool, FIFO_DEPTH> k_done("k_done");
    tapa::stream<bool, FIFO_DEPTH> v_done("v_done");
    tapa::streams<float, 2, FIFO_DEPTH> fifo_attention("fifo_attention");
    tapa::stream<int, FIFO_DEPTH> fifo_inst_softmax("fifo_inst_softmax");
    tapa::stream<float, FIFO_DEPTH> fifo_output("fifo_output");
    tapa::stream<float, FIFO_DEPTH> fifo_attn_out("fifo_attn_out");
    tapa::streams<float, 3, 4096> fifo_residual("fifo_residual");
    tapa::stream<float, FIFO_DEPTH> fifo_tok_to_rms("fifo_tok_to_rms");
    tapa::stream<float, FIFO_DEPTH> fifo_attn_to_rms("fifo_attn_to_rms");
    tapa::stream<float, FIFO_DEPTH> fifo_w1_to_silu("fifo_w1_to_silu");
    tapa::stream<float, FIFO_DEPTH> fifo_silu_to_elemul("fifo_silu_to_elemul");
    tapa::stream<float, FIFO_DEPTH> fifo_w3_to_elemul("fifo_w3_to_elemul");
    tapa::stream<float, FIFO_DEPTH> fifo_elemul_to_w2("fifo_elemul_to_w2");
    tapa::stream<float, FIFO_DEPTH> fifo_w2_exit("fifo_w2_exit");
    tapa::stream<float, FIFO_DEPTH> fifo_logits("fifo_logits");

    tapa::task()
        .invoke<tapa::join>(
            read_token_embedding,
            token_index,
            token_embedding_table,
            token_emb
        )
        // read weights
        .invoke<tapa::join>(read_weight, rms_att_weight, rms_att_w)
        .invoke<tapa::join>(read_weight, rms_ffn_weight, rms_ffn_w)
        .invoke<tapa::join>(read_final_rms_weight, rms_final_weight, rms_final_w)
        .invoke<tapa::join>(read_final_output_weight, output_weight, fifo_output_w)
        .invoke<tapa::join>(read_weight_kqv, wq, fifo_wq)
        .invoke<tapa::join>(read_weight_kqv, wk, fifo_wk)
        .invoke<tapa::join>(read_weight_kqv, wv, fifo_wv)
        .invoke<tapa::join>(read_weight_kqv, wo, fifo_wo)
        .invoke<tapa::join>(read_ffn_weight, w1, fifo_w1)
        .invoke<tapa::join>(read_ffn_weight, w2, fifo_w2)
        .invoke<tapa::join>(read_ffn_weight, w3, fifo_w3)
        // control layer and forwards
        .invoke<tapa::join>(
            layer_control_adapter,
            token_emb,
            fifo_layer_out,
            fifo_layer_in,
            fifo_exit
        )

        //start layer
        .invoke<tapa::detach>(duplicate, fifo_layer_in, fifo_tok_to_rms, fifo_residual)
        .invoke<tapa::detach>(
            rmsnorm,
            fifo_rms_to_kqv,
            fifo_tok_to_rms,
            rms_att_w
        )
        .invoke<tapa::join>(
            matmul_kqv,
            fifo_rms_to_kqv,
            fifo_rms_to_kqv,
            fifo_wq,
            fifo_query,
            DIM, DIM
        )
        .invoke<tapa::join>(
            matmul_kqv,
            fifo_rms_to_kqv,
            fifo_rms_to_kqv,
            fifo_wk,
            fifo_key,
            DIM, DIM
        )
        .invoke<tapa::join>(
            matmul_kqv,
            fifo_rms_to_kqv,
            fifo_rms_to_kqv,
            fifo_wv,
            fifo_value,
            DIM, DIM
        )
        .invoke<tapa::detach>(float_to_float_vec, fifo_query, fifo_query_v2)
        .invoke<tapa::detach>(float_to_float_vec, fifo_key, fifo_key_v2)
        .invoke<tapa::join>(read_freq_cis, token_position, freq_cis, fifo_freq_cis)
        .invoke<tapa::detach>(repeat_freq_cis, fifo_freq_cis, fifo_freq_cis)
        .invoke<tapa::detach>(
            RoPE,
            fifo_query_v2,
            fifo_key_v2,
            fifo_freq_cis,
            fifo_query_v2,
            fifo_key_v2
        )
        .invoke<tapa::detach>(float_vec_to_float, fifo_query_v2, fifo_query)
        .invoke<tapa::detach>(float_vec_to_float, fifo_key_v2, fifo_key)
        .invoke<tapa::detach>(black_hole_float, fifo_rms_to_kqv)
        .invoke<tapa::join>(kv_cache, token_position, fifo_key, key_cache, k_done)
        .invoke<tapa::join>(kv_cache, token_position, fifo_value, value_cache, v_done)
        .invoke<tapa::join>(read_cache, token_position, k_done, key_cache, fifo_key)
        .invoke<tapa::join>(read_cache, token_position, v_done, value_cache, fifo_value)
        .invoke<tapa::join>(matmul_attention, token_position, fifo_query, fifo_key, fifo_attention, fifo_inst_softmax)
        .invoke<tapa::detach>(softmax, fifo_attention, fifo_attention, fifo_inst_softmax)
        .invoke<tapa::join>(matmul_value, token_position, fifo_attention, fifo_value, fifo_output)
        .invoke<tapa::join>(matmul_output, fifo_output, fifo_wo, fifo_attn_out)
        .invoke<tapa::detach>(accum, fifo_residual, fifo_attn_out, fifo_residual)
        .invoke<tapa::detach>(duplicate, fifo_residual, fifo_residual, fifo_attn_to_rms)
        .invoke<tapa::detach>(rmsnorm, fifo_rms_to_ffn, fifo_attn_to_rms, rms_ffn_w)
        .invoke<tapa::join>(matmul_kqv, fifo_rms_to_ffn, fifo_rms_to_ffn, fifo_w1, fifo_w1_to_silu, DIM, HIDDEN_DIM)
        .invoke<tapa::detach>(silu, fifo_w1_to_silu, fifo_silu_to_elemul)
        .invoke<tapa::join>(matmul_kqv, fifo_rms_to_ffn, fifo_rms_to_ffn, fifo_w3, fifo_w3_to_elemul, DIM, HIDDEN_DIM)
        .invoke<tapa::detach>(black_hole_float, fifo_rms_to_ffn)
        .invoke<tapa::detach>(ele_mul, fifo_silu_to_elemul, fifo_w3_to_elemul, fifo_elemul_to_w2)
        .invoke<tapa::join>(matmul_w2, fifo_elemul_to_w2, fifo_w2, fifo_w2_exit)
        .invoke<tapa::detach>(accum, fifo_residual, fifo_w2_exit, fifo_layer_out)
        // final rms + ffn
        .invoke<tapa::detach>(rmsnorm, fifo_rms_to_output, fifo_exit, rms_final_w)
        .invoke<tapa::join>(matmul_logits, fifo_rms_to_output, fifo_output_w, fifo_logits)
        // write out logits
        .invoke<tapa::join>(write_logits, fifo_logits, output);
        
}