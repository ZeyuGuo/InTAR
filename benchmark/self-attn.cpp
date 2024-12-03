#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <bits/stdc++.h>

using namespace std;

typedef vector<vector<float>> Matrix;

// Helper function to perform matrix multiplication
void matMul(const Matrix& A, const Matrix& B, Matrix& result) {
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t common_dim = A[0].size();

    result.assign(rows, vector<float>(cols, 0));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < common_dim; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Helper function to apply softmax to a vector
void softmax(const vector<float>& input, vector<float>& output) {
    output.resize(input.size());
    float max_val = *max_element(input.begin(), input.end());
    float sum = 0.0;

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum;
    }
}

// Helper function to apply softmax row-wise on a matrix
void softmax(const Matrix& input, Matrix& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        softmax(input[i], output[i]);
    }
}

// Transpose a matrix
void transpose(const Matrix& A, Matrix& result) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    result.assign(cols, vector<float>(rows, 0));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = A[i][j];
        }
    }
}

// Self-attention computation
void selfAttention(const Matrix& input, const Matrix& Wq, const Matrix& Wk, const Matrix& Wv, Matrix& output) {
    // Step 1: Compute Query, Key, and Value matrices
    Matrix Q, K, V;
    matMul(input, Wq, Q);
    matMul(input, Wk, K);
    matMul(input, Wv, V);

    // Step 2: Compute scaled dot-product attention scores
    Matrix K_transposed;
    transpose(K, K_transposed);
    Matrix scores;
    matMul(Q, K_transposed, scores);

    float scale = sqrt(K[0].size());
    for (size_t i = 0; i < scores.size(); ++i) {
        for (size_t j = 0; j < scores[i].size(); ++j) {
            scores[i][j] /= scale;
        }
    }

    // Step 3: Apply softmax to get attention weights
    Matrix attention_weights;
    softmax(scores, attention_weights);

    // Step 4: Compute the output as weighted sum of Value matrix
    matMul(attention_weights, V, output);
}

int main() {
    // Example input and weight matrices
    Matrix input = {
        {1.0, 0.0, 1.0},
        {1.3, 2.0, 0.0},
        {1.0, 1.0, 0.0}
    };

    Matrix Wq = {
        {0.1, 0.2},
        {0.3, 0.4},
        {0.5, 0.6}
    };

    Matrix Wk = {
        {0.1, 0.3},
        {0.2, 0.4},
        {0.5, 0.6}
    };

    Matrix Wv = {
        {0.7, 0.8},
        {0.9, 1.0},
        {1.1, 1.2}
    };

    // Compute self-attention
    Matrix output;
    selfAttention(input, Wq, Wk, Wv, output);

    // Print the result
    cout << "Self-Attention Output:" << endl;
    for (const auto& row : output) {
        for (float value : row) {
            cout << value << " ";
        }
        cout << endl;
    }

    return 0;
}
