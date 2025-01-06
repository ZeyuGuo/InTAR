#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Function to initialize a 2D vector (matrix) with random values
void initialize_matrix(std::vector<std::vector<float>>& matrix, int rows, int cols) {
    matrix.resize(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // Random values between -1 and 1
        }
    }
}

// Function to initialize a 1D vector (bias) with random values
void initialize_vector(std::vector<float>& vec, int size) {
    vec.resize(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // Random values between -1 and 1
    }
}

// Matrix-matrix multiplication
std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    int rows = A.size();
    int cols = B[0].size();
    int inner_dim = B.size();
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < inner_dim; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Element-wise addition of a matrix and a bias vector
std::vector<std::vector<float>> add_bias(const std::vector<std::vector<float>>& matrix, const std::vector<float>& bias) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<float>> result = matrix;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] += bias[j];
        }
    }
    return result;
}

// Apply SiLU activation function to each element of a matrix
std::vector<std::vector<float>> apply_silu(const std::vector<std::vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<float>> result = matrix;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = matrix[i][j] / (1.0 + exp(-matrix[i][j]));
        }
    }
    return result;
}

// Element-wise addition of two matrices
std::vector<std::vector<float>> add_matrices(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

// Forward pass through the FFN layer
std::vector<std::vector<float>> ffn_layer(const std::vector<std::vector<float>>& input,
                                           const std::vector<std::vector<float>>& W_up,
                                           const std::vector<std::vector<float>>& W_gate,
                                           const std::vector<std::vector<float>>& W_down) {
    // Up network: Linear -> SiLU
    std::vector<std::vector<float>> up = matmul(input, W_up);
    up = apply_silu(up);

    // Gate network
    std::vector<std::vector<float>> gate = matmul(input, W_gate);

    // Element-wise addition
    std::vector<std::vector<float>> combined = add_matrices(up, gate);

    // Down projection
    std::vector<std::vector<float>> output = matmul(combined, W_down);

    return output;
}

int main() {
    // Seed random number generator
    srand(static_cast<unsigned>(time(0)));

    // Input and weight dimensions
    int batch_size = 8; // Number of input examples
    int input_dim = 4096;  // Input dimension
    int hidden_dim = 11008; // Hidden dimension

    // Initialize weights and biases
    std::vector<std::vector<float>> W_up, W_gate, W_down;

    initialize_matrix(W_up, input_dim, hidden_dim);
    initialize_matrix(W_gate, input_dim, hidden_dim);
    initialize_matrix(W_down, hidden_dim, input_dim);

    // Randomly initialize input matrix (batch of inputs)
    std::vector<std::vector<float>> input;
    initialize_matrix(input, batch_size, input_dim);

    // Forward pass
    std::vector<std::vector<float>> output = ffn_layer(input, W_up, W_gate, W_down);

    // Print a portion of the output for validation
    std::cout << "Output (first 2 rows):" << std::endl;
    for (int i = 0; i < 2; ++i) { // Print only the first 2 rows to avoid overwhelming the console
        for (float val : output[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
