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

void Llama2(
    const int token_index,
    const int token_position,
    tapa::mmap<float> token_embedding_table,
    // TODO: merge these weights as much as possible
    tapa::mmap<float> rms_att_weight,
    tapa::mmap<float> rms_ffn_weight,
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
    tapa::mmap<float> output
);

DEFINE_string(bitstream, "", "path to bitstream file");

struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

struct TransformerWeights {
    tensor2d token_embedding_table;  // [vocab_size, dim]
    // weights for rmsnorms
    tensor2d rms_att_weight;  // [layer, dim]
    tensor2d rms_ffn_weight;  // [layer, dim]
    // weights for attention matmuls
    tensor3d wq;  // [layer, dim, dim]
    tensor3d wk;  // [layer, dim, dim]
    tensor3d wv;  // [layer, dim, dim]
    tensor3d wo;  // [layer, dim, dim]
    // weights for ffn
    tensor3d w1;  // [layer, hidden_dim, dim]
    tensor3d w2;  // [layer, dim, hidden_dim]
    tensor3d w3;  // [layer, hidden_dim, dim]
    // final rmsnorm
    tensor1d rms_final_weight;  // [dim]
    // freq_cis for RoPE relatively positional embeddings
    tensor2d freq_cis_real;  // [seq_len, (dim/n_heads)/2]
    tensor2d freq_cis_imag;  // [seq_len, (dim/n_heads)/2]
    tensor2d output_weight;
};

struct RunState {
    // current wave of activations
    tensor1d x;  // activation at current time stamp [dim]
    tensor1d xb;  // same, but inside a residual branch [dim]
    tensor1d xb2;  // an additional buffer just for convenience [dim]
    tensor1d hb;  // buffer for hidden dimension in the ffn [hidden_dim]
    tensor1d hb2;  // another buffer for hidden dimension in the ffn [hidden_dim]
    tensor1d q;  // query [dim]
    tensor1d k;  // key [dim]
    tensor1d v;  // value [dim]
    tensor1d attention;  // buffer for scores/attention values [seq_len]
    tensor1d logits;  // buffer for logits [vocab_size]
    // kv cache
    tensor3d key_cache;  // [layer, seq_len, dim]
    tensor3d value_cache;  // [layer, seq_len, dim]
};

// --------------------------------------------------------------------------------------
// Tensor allocation and deallocation

void resize_state_tensors(RunState &state, Config &config) {
    tensor1d(config.dim).swap(state.x);
    tensor1d(config.dim).swap(state.xb);
    tensor1d(config.dim).swap(state.xb2);
    tensor1d(config.hidden_dim).swap(state.hb);
    tensor1d(config.hidden_dim).swap(state.hb2);
    tensor1d(config.dim).swap(state.q);
    tensor1d(config.dim).swap(state.k);
    tensor1d(config.dim).swap(state.v);
    tensor1d(config.seq_len).swap(state.attention);
    tensor1d(config.vocab_size).swap(state.logits);
    tensor3d(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim))).swap(state.key_cache);
    tensor3d(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim))).swap(state.value_cache);
}

void free_state_tensors(RunState &state) {
    state.x.clear();
    state.xb.clear();
    state.xb2.clear();
    state.hb.clear();
    state.hb2.clear();
    state.q.clear();
    state.k.clear();
    state.v.clear();
    state.attention.clear();
    state.logits.clear();
    state.key_cache.clear();
    state.value_cache.clear();
}

void resize_weights_tensors(TransformerWeights &weights, Config &config) {
    tensor2d t;
    std::cout << "max size: " <<  t.max_size() << std::endl;
    std::cout << "alloc: " << config.vocab_size * config.dim << std::endl;
    tensor2d(config.vocab_size, tensor1d(config.dim)).swap(weights.token_embedding_table);
    std::cout << "start resize" << std::endl;
    tensor2d(config.n_layers, tensor1d(config.dim)).swap(weights.rms_att_weight);
    tensor2d(config.n_layers, tensor1d(config.dim)).swap(weights.rms_ffn_weight);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wq);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wk);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wv);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.dim))).swap(weights.wo);
    tensor3d(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim))).swap(weights.w1);
    tensor3d(config.n_layers, tensor2d(config.dim, tensor1d(config.hidden_dim))).swap(weights.w2);
    tensor3d(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim))).swap(weights.w3);
    tensor1d(config.dim).swap(weights.rms_final_weight);
    int head_size = config.dim / config.n_heads;
    tensor2d(config.seq_len, tensor1d(head_size / 2)).swap(weights.freq_cis_real);
    tensor2d(config.seq_len, tensor1d(head_size / 2)).swap(weights.freq_cis_imag);
    tensor2d(config.vocab_size, tensor1d(config.dim)).swap(weights.output_weight);
}

void free_weights_tensors(TransformerWeights &weights) {
    weights.token_embedding_table.clear();
    weights.rms_att_weight.clear();
    weights.rms_ffn_weight.clear();
    weights.wq.clear();
    weights.wk.clear();
    weights.wv.clear();
    weights.wo.clear();
    weights.w1.clear();
    weights.w2.clear();
    weights.w3.clear();
    weights.rms_final_weight.clear();
    weights.freq_cis_real.clear();
    weights.freq_cis_imag.clear();
    weights.output_weight.clear();
}

// --------------------------------------------------------------------------------------
// Initialization: random init or read from checkpoint


// TODO: merge these into one function
void checkpoint_init_tensor(tensor1d &tensor, std::fstream &file) {
    file.read((char*)(tensor.data()), tensor.size() * sizeof(float));
}
void checkpoint_init_tensor(tensor2d &tensor, std::fstream &file) {
    for (auto &t : tensor) checkpoint_init_tensor(t, file);
}
void checkpoint_init_tensor(tensor3d &tensor, std::fstream &file) {
    for (auto &t : tensor) checkpoint_init_tensor(t, file);
}

void checkpoint_init_weights(TransformerWeights &weights, Config &config, std::fstream &file) {
    checkpoint_init_tensor(weights.token_embedding_table, file);
    checkpoint_init_tensor(weights.rms_att_weight, file);
    checkpoint_init_tensor(weights.wq, file);
    checkpoint_init_tensor(weights.wk, file);
    checkpoint_init_tensor(weights.wv, file);
    checkpoint_init_tensor(weights.wo, file);
    checkpoint_init_tensor(weights.rms_ffn_weight, file);
    checkpoint_init_tensor(weights.w1, file);
    checkpoint_init_tensor(weights.w2, file);
    checkpoint_init_tensor(weights.w3, file);
    checkpoint_init_tensor(weights.rms_final_weight, file);
    checkpoint_init_tensor(weights.freq_cis_real, file);
    checkpoint_init_tensor(weights.freq_cis_imag, file);
    checkpoint_init_tensor(weights.output_weight, file);
}

void flatten(
    const tensor2d& origin,
    tensor1d& flat
){
    flat.clear();
    for(const auto &v : origin){
        flat.insert(flat.end(), v.begin(), v.end());
    }
}

void flatten3d(
    const tensor3d& origin,
    tensor1d& flat
){
    flat.clear();
    for(const auto &v2d: origin){
        for(const auto &v : v2d){
            flat.insert(flat.end(), v.begin(), v.end());
        }
    }
}

void merge_vec(
    const tensor2d& x,
    const tensor2d& y,
    tensor1d& output
){
    for(int l = 0; l < x.size(); l++){
        for(int i = 0; i < x[0].size(); i++){
            output.push_back(x[l][i]);
            output.push_back(y[l][i]);
        }
    }
}


int main(int argc, char *argv[]) {
    // std::cout.tie(NULL);

    std::string checkpoint;
    float temperature = 0.6;
    // 'checkpoint' is a required arg
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <checkpoint_file> [temperature]\n";
        return 1;
    }
    checkpoint = argv[1];
    // temperature is optional
    if (argc >= 3)
        temperature = std::atof(argv[2]);

    Config config;
    TransformerWeights transformer_weights;
    {
        std::fstream file(checkpoint);
        if (!file) {
            std::cout << "Unable to open the checkpoint file " << checkpoint << "\n";
            return 1;
        }
        // read file contents to config
        file.read((char*)&config, sizeof(config));
        config.seq_len = 512;
        resize_weights_tensors(transformer_weights, config);
        checkpoint_init_weights(transformer_weights, config, file);
        file.close();
    }

    std::vector<std::string> vocab(config.vocab_size);
    {
        std::fstream file("tokenizer.bin");
        if (!file) {
            std::cout
                << "Unable to open the tokenizer file tokenizer.bin! Run \n"
                << "python tokenizer.py to convert tokenizer.model -> tokenizer.bin\n";
            return 1;
        }
        for (int i = 0; i < config.vocab_size; i++) {
            int len;
            vocab[i] = "";
            file.read((char*)&len, sizeof(int));
            for (int j = 0; j < len; ++j) {
                char c;
                file.read((char*)&c, sizeof(char));
                vocab[i].push_back(c);
            }
            vocab[i].push_back('\0');
        }
        file.close();
    }

    std::cout << "finish loading tokenizer." << std::endl;

    RunState state;
    resize_state_tensors(state, config);

    // prompts
    std::vector<int> prompts = {1,  3439, 17632,  1925, 29892,   278,  6368,   310, 14215,   537,
          5922,   393, 29871};
    // for(int i = 0; i < prompts.size(); i++){
    //     std::cout << vocab[prompts[i]];
    //     std::cout.flush();
    // }

    clock_t start = clock();
    int next;
    int token = prompts[0];  // 1 = BOS token in Llama-2 sentence-piece

    tensor1d token_emb_fpga;
    tensor1d rms_att_w_fpga; tensor1d rms_ffn_w_fpga;
    tensor1d wq_fpga; tensor1d wk_fpga; tensor1d wv_fpga; tensor1d wo_fpga;
    tensor1d w1_fpga; tensor1d w2_fpga; tensor1d w3_fpga;
    tensor1d freq_cis;
    tensor1d output(4096, 0);
    tensor1d key_cache(4096*512*32, 0);
    tensor1d value_cache(4096*512*32, 0);
    flatten(transformer_weights.token_embedding_table, token_emb_fpga);
    flatten(transformer_weights.rms_att_weight, rms_att_w_fpga);
    flatten(transformer_weights.rms_ffn_weight, rms_ffn_w_fpga);
    flatten3d(transformer_weights.wq, wq_fpga);
    flatten3d(transformer_weights.wk, wk_fpga);
    flatten3d(transformer_weights.wv, wv_fpga);
    flatten3d(transformer_weights.wo, wo_fpga);
    flatten3d(transformer_weights.w1, w1_fpga);
    flatten3d(transformer_weights.w2, w2_fpga);
    flatten3d(transformer_weights.w3, w3_fpga);
    merge_vec(transformer_weights.freq_cis_real, transformer_weights.freq_cis_imag, freq_cis);


    int64_t kernel_time_ns = tapa::invoke(Llama2, FLAGS_bitstream,
                        token, 0,
                        tapa::read_only_mmap<float>(token_emb_fpga),
                        tapa::read_only_mmap<float>(rms_att_w_fpga),
                        tapa::read_only_mmap<float>(rms_ffn_w_fpga),
                        tapa::read_only_mmap<float>(wq_fpga),
                        tapa::read_only_mmap<float>(wk_fpga),
                        tapa::read_only_mmap<float>(wv_fpga),
                        tapa::read_only_mmap<float>(wo_fpga),
                        tapa::read_only_mmap<float>(w1_fpga),
                        tapa::read_only_mmap<float>(w2_fpga),
                        tapa::read_only_mmap<float>(w3_fpga),
                        tapa::read_only_mmap<float>(freq_cis).reinterpret<float_v2>(),
                        tapa::read_write_mmap<float>(key_cache),
                        tapa::read_write_mmap<float>(value_cache),
                        tapa::write_only_mmap<float>(output)
                        );
    // for (int pos = 0; pos < config.seq_len; ++pos) {
    //     // forward the transformer to get logits for the next token
    //     transformer(token, pos, config, state, transformer_weights);
    //     if (temperature < EPS) {
    //         next = argmax(state.logits);
    //     } else {
    //         for (int q = 0; q < config.vocab_size; ++q)
    //             state.logits[q] /= temperature;
    //         softmax(state.logits, state.logits);
    //         next = sample(state.logits);
    //     }
    //     if(pos < prompts.size()-1){
    //         token = prompts[pos+1];
    //     } else {
    //         std::cout << vocab[next];
    //         std::cout.flush();
    //         token = next;
    //     }
    // }
    // std::cout << "\n";

    // report our achieved tok/s
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("achieved tok/s: %f\n", config.seq_len / elapsed);

    // memory cleanup
    free_state_tensors(state);
    free_weights_tensors(transformer_weights);
    // vocab.clear();

    return 0;
}