#include <iostream>
#include <vector>
#include <cmath>

constexpr int VOCAB_SIZE = 32000;
constexpr int DIM = 4096;

void parallel_embedding(
    const int vocab_size,
    const int dim,
    const std::vector<std::vector<float>>& weight,
    const std::vector<int>& tokens,
    std::vector<std::vector<float>>& res
){
    res.clear();
    for(int i : tokens){
        res.push_back(weight[i]);
    }
}

void linear(
    const int dim,
    const int hidden_dim,
    const std::vector<std::vector<float>>& weight,
    const std::vector<std::vector<float>>& input,
    std::vector<std::vector<float>>& output
){
    const int width = input.size();

    for(int i = 0; i < width; i++){
        for(int j = 0; j < hidden_dim; j++){
            for(int k = 0; k < dim; k++){
                output[i][j] += input[i][k] * weight[k][j];
            }
        }
    }
}

void silu_linear(
    const int dim,
    const int hidden_dim,
    const std::vector<std::vector<float>>& weight,
    const std::vector<std::vector<float>>& input,
    std::vector<std::vector<float>>& output
){
    const int width = input.size();

    for(int i = 0; i < width; i++){
        for(int j = 0; j < hidden_dim; j++){
            for(int k = 0; k < dim; k++){
                output[i][j] += input[i][k] * weight[k][j];
            }
            output[i][j] = silu(output[i][j]);
        }
    }
}

float silu(const float x){
    float output = x / (1.0 - std::exp(x));
    return output;
}

void ffn(
    const int dim,
    const int hidden_dim,
    const std::vector<std::vector<float>>& w1,
    const std::vector<std::vector<float>>& w2,
    const std::vector<std::vector<float>>& w3,
    const std::vector<std::vector<float>>& input,
    std::vector<std::vector<float>>& output
){
    const int width = input.size();

    std::vector<std::vector<float>> w1_o(width, std::vector<float>(hidden_dim, 0.0));
    std::vector<std::vector<float>> w3_o(width, std::vector<float>(hidden_dim, 0.0));
    std::vector<std::vector<float>> w2_o(width, std::vector<float>(dim, 0.0));

    silu_linear(dim, hidden_dim, w1, input, w1_o);
    linear(dim, hidden_dim, w3, input, w3_o);
    for(int i = 0; i < width; i++){
        for(int j = 0; j < hidden_dim; j++){
            w1_o[i][j] *= w3_o[i][j];
        }
    }

    linear(hidden_dim, dim, w2, w1_o, output);
}

void rms_norm(
    const int dim,
    const float eps,
    const std::vector<float>& weight,
    const std::vector<std::vector<float>>& input,
    std::vector<std::vector<float>>& output
){
    const int depth = input.size();
    for(int i = 0; i < depth; i++){
        float cache_i[dim];

        for(int j = 0; j < dim; j++){
            
        }
    }
}

void transformer(
    const std::vector<std::vector<int>>& tokens,
    const std::vector<std::vector<float>>& embedding_weight, // params for embedding
    const int start_pos,
    std::vector<std::vector<std::vector<float>>>& output
){
    const int batch_size = tokens.size();
    const int seq_len = tokens[0].size();

    std::vector<std::vector<float>> h;
    parallel_embedding(VOCAB_SIZE, DIM, embedding_weight, tokens, h);


}

int main(int argc, char* argv[]){

    // read weights and input for ffn

    // call ffn

    return 0;
}