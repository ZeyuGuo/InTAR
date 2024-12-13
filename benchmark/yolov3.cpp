#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <bits/stdc++.h>

// Constants
const int INPUT_SIZE = 16;  // Example input size for testing

// Function to initialize convolution weights
void initialize_conv_weights(std::vector<std::vector<std::vector<std::vector<float>>>> &weights,
                             int num_filters, int channels, int kernel_size) {
    weights.resize(num_filters, std::vector<std::vector<std::vector<float>>>(
                                    channels, std::vector<std::vector<float>>(
                                                 kernel_size, std::vector<float>(kernel_size, 0.1f))));
}

// Function to initialize weights for a fully connected layer
void initialize_fc_weights(std::vector<std::vector<float>> &weights, int input_size, int output_size) {
    weights.resize(output_size, std::vector<float>(input_size, 0.1f)); // Initialize with small constant value
}

// Convolution layer
void convolution_layer(const std::vector<std::vector<std::vector<float>>> &input,
                       std::vector<std::vector<std::vector<float>>> &output,
                       const std::vector<std::vector<std::vector<std::vector<float>>>> &weights,
                       const std::vector<float> &bias,
                       int stride, int kernel_size, int padding_size = 0) {
    int input_channels = input.size();
    int output_channels = weights.size();
    int output_size = (input[0].size() + 2 * padding_size - kernel_size) / stride + 1;

    output.resize(output_channels, std::vector<std::vector<float>>(
                                      output_size, std::vector<float>(output_size, 0.0f)));

    for (int oc = 0; oc < output_channels; ++oc) { // Loop over output channels
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                float sum = 0.0f;
                for (int ic = 0; ic < input_channels; ++ic) { // Loop over input channels
                    for (int ki = 0; ki < kernel_size; ++ki) { // Kernel dimension
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int x = i * stride + ki;
                            int y = j * stride + kj;
                            auto op1 = (x < input[0].size() && y < input[0].size()) ? input[ic][x][y] : 0.f;
                            sum += op1 * weights[oc][ic][ki][kj];
                        }
                    }
                }
                output[oc][i][j] = sum + bias[oc];
            }
        }
    }
}

// Average pooling layer
void average_pool_layer(const std::vector<std::vector<std::vector<float>>> &input,
                        std::vector<std::vector<std::vector<float>>> &output,
                        int pool_size, int stride) {
    int channels = input.size();
    int input_size = input[0].size();
    int output_size = (input_size - pool_size) / stride + 1;

    output.resize(channels, std::vector<std::vector<float>>(
                               output_size, std::vector<float>(output_size, 0.0f)));

    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                float sum = 0.0f;
                for (int pi = 0; pi < pool_size; ++pi) {
                    for (int pj = 0; pj < pool_size; ++pj) {
                        int x = i * stride + pi;
                        int y = j * stride + pj;
                        sum += input[c][x][y];
                    }
                }
                output[c][i][j] = sum / (pool_size * pool_size);
            }
        }
    }
}

// Fully connected layer
void fully_connected_layer(const std::vector<float> &input,
                           std::vector<float> &output,
                           const std::vector<std::vector<float>> &weights,
                           const std::vector<float> &bias) {
    int input_size = input.size();
    int output_size = weights.size();

    output.resize(output_size, 0.0f);

    for (int i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; ++j) {
            sum += input[j] * weights[i][j];
        }
        output[i] = sum + bias[i];
    }
}

// Softmax function
void softmax(std::vector<float> &input) {
    float max_val = *max_element(input.begin(), input.end());
    float sum = 0.0f;

    for (float &val : input) {
        val = exp(val - max_val); // Subtract max for numerical stability
        sum += val;
    }

    for (float &val : input) {
        val /= sum;
    }
}

void leakyReLU(std::vector<std::vector<std::vector<float>>>& vec, float alpha = 0.01) {
    // Traverse each element in the 3D vector
    for (auto& matrix : vec) {
        for (auto& row : matrix) {
            for (auto& value : row) {
                // Apply Leaky ReLU
                if (value < 0) {
                    value *= alpha;
                }
            }
        }
    }
}

void residual(std::vector<std::vector<std::vector<float>>>& vec1, std::vector<std::vector<std::vector<float>>>& vec2, std::vector<std::vector<std::vector<float>>>& output){
    int i_bound = vec1.size();
    int j_bound = vec1[0].size();
    int k_bound = vec1[0][0].size();

    output.resize(i_bound, std::vector<std::vector<float>>(j_bound, std::vector<float>(k_bound, 0.f)));

    for(int i = 0; i < i_bound; i++){
        for(int j = 0; j < j_bound; j++){
            for(int k = 0; k < k_bound; k++){
                output[i][j][k] = vec1[i][j][k] + vec2[i][j][k];
            }
        }
    }
}

void YOLOv3_last(
    std::vector<std::vector<std::vector<float>>>& input,
    std::vector<std::vector<std::vector<std::vector<float>>>>& conv_weights1,
    std::vector<std::vector<std::vector<std::vector<float>>>>& conv_weights2,
    std::vector<std::vector<std::vector<std::vector<float>>>>& conv_weights3,
    std::vector<std::vector<float>>& fc_weights,
    std::vector<float>& fc_bias,
    std::vector<float>& conv_bias1,
    std::vector<float>& conv_bias2,
    std::vector<float>& output
){

    std::vector<std::vector<std::vector<float>>> conv_output1;
    std::vector<std::vector<std::vector<float>>> conv_output2;
    std::vector<std::vector<std::vector<float>>> conv_output3;
    std::vector<std::vector<std::vector<float>>> res_output;

    convolution_layer(input, conv_output1, conv_weights1, conv_bias1, 2, 3, 1); // Stride 1, kernel size 3x3
    leakyReLU(conv_output1);

    for(int i = 0; i < 4; i++){
        convolution_layer(conv_output1, conv_output2, conv_weights2, conv_bias2, 1, 1); // Stride 1, kernel size 3x3
        leakyReLU(conv_output2);
        convolution_layer(conv_output2, conv_output3, conv_weights3, conv_bias1, 1, 3, 1); // Stride 1, kernel size 3x3
        leakyReLU(conv_output3);
        residual(conv_output3, conv_output1, res_output); 

        std::cout << "size after conv1: " << conv_output1.size() * conv_output1[0].size() * conv_output1[0][0].size() << std::endl;
        std::cout << "size after conv2: " << conv_output2.size() * conv_output2[0].size() * conv_output2[0][0].size() << std::endl;
        std::cout << "size after conv3: " << conv_output3.size() * conv_output3[0].size() * conv_output3[0][0].size() << std::endl;

        conv_output1 = res_output;
    }

    // Average pooling layer
    std::vector<std::vector<std::vector<float>>> avg_pool_output;
    average_pool_layer(res_output, avg_pool_output, 8, 8); // Pool size 2x2, stride 2

    std::cout << "size after avgpool: " << avg_pool_output.size() * avg_pool_output[0].size() * avg_pool_output[0][0].size() << std::endl; 

    // Flatten the average pool output for the fully connected layer
    std::vector<float> fc_input;
    for (const auto &channel : avg_pool_output) {
        for (const auto &row : channel) {
            for (float val : row) {
                fc_input.push_back(val);
            }
        }
    }

    // Fully connected layer
    std::vector<float> fc_output;
    fully_connected_layer(fc_input, fc_output, fc_weights, fc_bias);

    // Apply softmax activation
    softmax(fc_output);
    output = fc_output;
}

// Example main function to demonstrate the pipeline
int main() {
    // Example input (3 channels, 8x8)
    std::vector<std::vector<std::vector<float>>> input(512, std::vector<std::vector<float>>(
                                                           INPUT_SIZE, std::vector<float>(INPUT_SIZE, 1.0f)));

    // Convolution layer
    std::vector<std::vector<std::vector<std::vector<float>>>> conv_weights1;
    std::vector<std::vector<std::vector<std::vector<float>>>> conv_weights2;
    std::vector<std::vector<std::vector<std::vector<float>>>> conv_weights3;
    std::vector<float> conv_bias1(1024, 0.1f); // 1024 filters
    std::vector<float> conv_bias2(512, 0.1f); // 512 filters
    initialize_conv_weights(conv_weights1, 1024, 512, 3); // 8 filters, 3 input channels, kernel size 3x3
    initialize_conv_weights(conv_weights2, 512, 1024, 1);
    initialize_conv_weights(conv_weights3, 1024, 512, 3);

    std::vector<std::vector<float>> fc_weights;
    std::vector<float> fc_bias(1000, 0.1f); // 10 output neurons
    initialize_fc_weights(fc_weights, 1024, 1000);

    std::vector<float> output;

    YOLOv3_last(input, conv_weights1, conv_weights2, conv_weights3, fc_weights, fc_bias, conv_bias1, conv_bias2, output);

    //Print softmax output
    // std::cout << "Softmax Output:" << std::endl;
    // for (float val : fc_output) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

    return 0;
}
