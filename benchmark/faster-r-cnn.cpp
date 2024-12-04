#include <iostream>
#include <vector>
#include <cmath>

constexpr int image_size = 14;

// Function for 2D convolution
void conv2D(const std::vector<std::vector<float>>& input,
            const std::vector<std::vector<float>>& kernel,
            std::vector<std::vector<float>>& output) {
    int inputSize = input.size();
    int kernelSize = kernel.size();
    int outputSize = inputSize - kernelSize + 1;

    output.resize(outputSize, std::vector<float>(outputSize, 0));

    // Kernel dimension: kernelSize x kernelSize
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    output[i][j] += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
        }
    }
}

void initialize_conv_weights(std::vector<std::vector<std::vector<std::vector<float>>>> &weights,
                             int num_filters, int channels, int kernel_size) {
    weights.resize(num_filters, std::vector<std::vector<std::vector<float>>>(
                                    channels, std::vector<std::vector<float>>(
                                                 kernel_size, std::vector<float>(kernel_size, 0.1f))));
}

void convolution_layer(const std::vector<std::vector<std::vector<float>>> &input,
                       std::vector<std::vector<std::vector<float>>> &output,
                       const std::vector<std::vector<std::vector<std::vector<float>>>> &weights,
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
                output[oc][i][j] = sum;
            }
        }
    }
}

// Function to apply ReLU activation
void relu(std::vector<std::vector<std::vector<float>>>& input) {
    for (auto& channel : input){
        for (auto& row : channel) {
            for (auto& val : row) {
                val = std::max(0.0f, val);
            }
        }
    }
}

// Function to perform max pooling
void maxPool(const std::vector<std::vector<float>>& input,
             std::vector<std::vector<float>>& output,
             int poolSize) {
    int inputSize = input.size();
    int outputSize = inputSize / poolSize;

    output.resize(outputSize, std::vector<float>(outputSize, 0));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float maxVal = -INFINITY;
            for (int pi = 0; pi < poolSize; ++pi) {
                for (int pj = 0; pj < poolSize; ++pj) {
                    maxVal = std::max(maxVal, input[i * poolSize + pi][j * poolSize + pj]);
                }
            }
            output[i][j] = maxVal;
        }
    }
}

// Simplified Region Proposal
void regionProposal(const std::vector<std::vector<float>>& input,
                    std::vector<std::pair<int, int>>& proposals) {
    int threshold = 5; // Example threshold for proposals
    for (int i = 0; i < input.size(); ++i) {
        for (int j = 0; j < input[i].size(); ++j) {
            if (input[i][j] > threshold) {
                proposals.emplace_back(i, j);
            }
        }
    }
}

// Basic classifier
void classifyRegions(const std::vector<std::pair<int, int>>& proposals,
                     std::vector<int>& classes) {
    for (const auto& proposal : proposals) {
        // Simplified classification logic
        classes.push_back(proposal.first % 2); // Example: classify as class 0 or 1
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

void Faster_R_CNN(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<std::vector<std::vector<std::vector<float>>>> &conv1_weights, //7x7
    const std::vector<std::vector<std::vector<std::vector<float>>>> &conv2_weights, //1x1
    const std::vector<std::vector<std::vector<std::vector<float>>>> &conv3_weights, //3x3
    std::vector<std::vector<int>>& classes
){
    std::vector<std::vector<std::vector<float>>> conv1_output;
    convolution_layer(input, conv1_output, conv1_weights, 2, 7, 4);
    relu(conv1_output);

    std::vector<std::vector<std::vector<float>>> conv2_output;
    std::vector<std::vector<std::vector<float>>> conv3_output;
    std::vector<std::vector<std::vector<float>>> residual_output;
    convolution_layer(conv1_output, conv2_output, conv2_weights, 1, 1, 0);
    relu(conv2_output);
    convolution_layer(conv2_output, conv3_output, conv3_weights, 1, 3, 1);
    relu(conv3_output);
    residual(conv1_output, conv3_output, residual_output);
    conv1_output = residual_output;

    for(auto& channel : conv1_output){
        std::vector<std::pair<int, int>> proposal;
        std::vector<int> c_class;
        regionProposal(channel, proposal);
        classifyRegions(proposal, c_class);
        classes.push_back(c_class);
    }

}

int main() {
    // Example input (3 channels, 8x8)
    std::vector<std::vector<std::vector<float>>> input(1024, std::vector<std::vector<float>>(
                                                           image_size, std::vector<float>(image_size, 1.0f)));

    // Convolution layer
    std::vector<std::vector<std::vector<std::vector<float>>>> conv_weights1;
    std::vector<std::vector<std::vector<std::vector<float>>>> conv_weights2;
    std::vector<std::vector<std::vector<std::vector<float>>>> conv_weights3;
    initialize_conv_weights(conv_weights1, 2048, 1024, 7); // 8 filters, 3 input channels, kernel size 3x3
    initialize_conv_weights(conv_weights2, 1024, 2048, 1);
    initialize_conv_weights(conv_weights3, 2048, 1024, 3);

    std::vector<std::vector<int>> classes;

    Faster_R_CNN(input, conv_weights1, conv_weights2, conv_weights3, classes);

    return 0;
}
