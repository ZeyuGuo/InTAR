#include <iostream>
#include <vector>
#include <cmath>

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

// Function to apply ReLU activation
void relu(std::vector<std::vector<float>>& input) {
    for (auto& row : input) {
        for (auto& val : row) {
            val = std::max(0.0f, val);
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

int main() {
    // Example input image (5x5)
    std::vector<std::vector<float>> input = {
        {1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1},
        {1, 3, 5, 3, 1},
        {1, 2, 1, 2, 1},
        {3, 4, 5, 4, 3}
    };

    // Example kernel (3x3)
    std::vector<std::vector<float>> kernel = {
        {1, 2, -1},
        {1, 3, -1},
        {1, 1, -1}
    };

    // Convolution output
    std::vector<std::vector<float>> convOutput;
    conv2D(input, kernel, convOutput);

    // Apply ReLU
    relu(convOutput);

    // Max pooling with pool size 2
    std::vector<std::vector<float>> pooledOutput;
    maxPool(convOutput, pooledOutput, 2);

    // Region proposals
    std::vector<std::pair<int, int>> proposals;
    regionProposal(pooledOutput, proposals);

    // Classification
    std::vector<int> classes;
    classifyRegions(proposals, classes);

    // Output results
    std::cout << "Region Proposals:" << std::endl;
    for (const auto& proposal : proposals) {
        std::cout << "(" << proposal.first << ", " << proposal.second << ")" << std::endl;
    }

    std::cout << "Classes:" << std::endl;
    for (int cls : classes) {
        std::cout << cls << std::endl;
    }

    return 0;
}
