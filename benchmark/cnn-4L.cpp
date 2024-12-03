#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// Helper typedef for 2D matrix
typedef std::vector<std::vector<float>> Matrix;

// Function to print a matrix (for debugging)
void printMatrix(const Matrix& mat) {
    for (const auto& row : mat) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// Perform 2D convolution
void convolve(const Matrix& input, const Matrix& kernel, Matrix& output) {
    int inputSize = input.size();
    int kernelSize = kernel.size();
    int outputSize = (input[0].size() - kernelSize) + 1;

    output.resize(outputSize, std::vector<float>(outputSize, 0));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            for (int m = 0; m < kernelSize; ++m) {
                for (int n = 0; n < kernelSize; ++n) {
                    output[i][j] += input[i + m][j + n] * kernel[m][n];
                }
            }
        }
    }
}

// Apply ReLU activation
void relu(const Matrix& input, Matrix& output) {
    output = input;
    for (auto& row : output) {
        for (auto& val : row) {
            val = std::max(0.0f, val);
        }
    }
}

// Upsample by a factor of 2
void upsample(const Matrix& input, Matrix& output) {
    int inputSize = input.size();
    int outputSize = inputSize * 2;
    output.resize(outputSize, std::vector<float>(outputSize, 0));

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            output[2 * i][2 * j] = input[i][j];
            output[2 * i + 1][2 * j] = input[i][j];
            output[2 * i][2 * j + 1] = input[i][j];
            output[2 * i + 1][2 * j + 1] = input[i][j];
        }
    }
}

// Max pooling with 2x2 filter and stride 2
void maxPool(const Matrix& input, Matrix& output) {
    int inputSize = input.size();
    int outputSize = inputSize / 2;
    output.resize(outputSize, std::vector<float>(outputSize, 0));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float maxVal = 0;
            for (int m = 0; m < 2; ++m) {
                for (int n = 0; n < 2; ++n) {
                    maxVal = std::max(maxVal, input[2 * i + m][2 * j + n]);
                }
            }
            output[i][j] = maxVal;
        }
    }
}

// Main function demonstrating the 4-layer CNN
int main() {
    // Example input matrix (8x8)
    Matrix input = {
        {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
        {5, 6, 7, 8, 1, 2, 3, 4, 1, 2, 3, 4},
        {9, 10, 11, 12, 1, 2, 3, 4, 1, 2, 3, 4},
        {13, 14, 15, 16, 1, 2, 3, 4, 1, 2, 3, 4},
        {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
        {5, 6, 7, 8, 1, 2, 3, 4, 1, 2, 3, 4},
        {9, 10, 11, 12, 1, 2, 3, 4, 1, 2, 3, 4},
        {13, 14, 15, 16, 1, 2, 3, 4,1, 2, 3, 4},
        {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
        {5, 6, 7, 8, 1, 2, 3, 4, 1, 2, 3, 4},
        {9, 10, 11, 12, 1, 2, 3, 4, 1, 2, 3, 4},
        {13, 14, 15, 16, 1, 2, 3, 4,1, 2, 3, 4}
    };

    // Example kernel (3x3)
    Matrix kernel = {
        {1, 1, -1},
        {1, 2, -1},
        {1, 3, -1}
    };

    Matrix layer1, layer2, layer3, layer4, temp1, temp2, temp3, temp4;

    // Layer 1: Convolution + ReLU + Upsampling
    convolve(input, kernel, temp1);
    relu(temp1, temp1);
    upsample(temp1, layer1);
    std::cout << "Layer 1 (Upsample):" << std::endl;
    printMatrix(layer1);

    // Layer 2: Convolution + ReLU + Upsampling
    convolve(layer1, kernel, temp2);
    relu(temp2, temp2);
    upsample(temp2, layer2);
    std::cout << "\nLayer 2 (Upsample):" << std::endl;
    printMatrix(layer2);

    // Layer 3: Convolution + ReLU + Max Pooling
    convolve(layer2, kernel, temp3);
    relu(temp3, temp3);
    maxPool(temp3, layer3);
    std::cout << "\nLayer 3 (Downsample):" << std::endl;
    printMatrix(layer3);

    // Layer 4: Convolution + ReLU + Max Pooling
    convolve(layer3, kernel, temp4);
    relu(temp4, temp4);
    maxPool(temp4, layer4);
    std::cout << "\nLayer 4 (Downsample):" << std::endl;
    printMatrix(layer4);

    return 0;
}
