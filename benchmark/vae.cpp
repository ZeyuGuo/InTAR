#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Hyperparameters
const int input_dim = 8; // Input size (1D for simplicity)
const int hidden_dim1 = 4; // Number of neurons in the first hidden layer
const int hidden_dim2 = 2; // Number of neurons in the second hidden layer
const int latent_dim = 2;  // Latent space dimensionality
const int output_dim = input_dim;

// Activation function (ReLU)
void relu(float* input, int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = std::max(0.0f, input[i]);
    }
}

// Sigmoid activation function
void sigmoid(float* input, int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

// Encoder: Input -> Hidden layer 1 -> Hidden layer 2 -> Latent mean and log variance
void encoder(float* input, float* latent_mean, float* latent_log_var) {
    float hidden1[hidden_dim1] = {0}; // First hidden layer
    float hidden2[hidden_dim2] = {0}; // Second hidden layer

    // First layer: input_dim -> hidden_dim1
    for (int i = 0; i < hidden_dim1; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            hidden1[i] += input[j] * 0.01f; // Kernel: [input_dim x hidden_dim1], weights initialized as 0.01
        }
        hidden1[i] += 0.1f; // Bias term
    }
    relu(hidden1, hidden_dim1);

    // Second layer: hidden_dim1 -> hidden_dim2
    for (int i = 0; i < hidden_dim2; ++i) {
        for (int j = 0; j < hidden_dim1; ++j) {
            hidden2[i] += hidden1[j] * 0.01f; // Kernel: [hidden_dim1 x hidden_dim2]
        }
        hidden2[i] += 0.1f; // Bias term
    }
    relu(hidden2, hidden_dim2);

    // Latent space: hidden_dim2 -> latent_dim
    for (int i = 0; i < latent_dim; ++i) {
        for (int j = 0; j < hidden_dim2; ++j) {
            latent_mean[i] += hidden2[j] * 0.01f; // Kernel for mean
            latent_log_var[i] += hidden2[j] * 0.01f; // Kernel for log variance
        }
    }
}

// Decoder: Latent -> Hidden layer 2 -> Hidden layer 1 -> Output
void decoder(float* latent_sample, float* output) {
    float hidden2[hidden_dim2] = {0}; // First hidden layer (decoder)
    float hidden1[hidden_dim1] = {0}; // Second hidden layer (decoder)

    // First layer: latent_dim -> hidden_dim2
    for (int i = 0; i < hidden_dim2; ++i) {
        for (int j = 0; j < latent_dim; ++j) {
            hidden2[i] += latent_sample[j] * 0.01f; // Kernel: [latent_dim x hidden_dim2]
        }
        hidden2[i] += 0.1f; // Bias term
    }
    relu(hidden2, hidden_dim2);

    // Second layer: hidden_dim2 -> hidden_dim1
    for (int i = 0; i < hidden_dim1; ++i) {
        for (int j = 0; j < hidden_dim2; ++j) {
            hidden1[i] += hidden2[j] * 0.01f; // Kernel: [hidden_dim2 x hidden_dim1]
        }
        hidden1[i] += 0.1f; // Bias term
    }
    relu(hidden1, hidden_dim1);

    // Output layer: hidden_dim1 -> output_dim
    for (int i = 0; i < output_dim; ++i) {
        for (int j = 0; j < hidden_dim1; ++j) {
            output[i] += hidden1[j] * 0.01f; // Kernel: [hidden_dim1 x output_dim]
        }
        output[i] += 0.1f; // Bias term
    }
    sigmoid(output, output_dim);
}

// Sampling latent vector using reparameterization trick
void sample_latent(float* latent_mean, float* latent_log_var, float* latent_sample) {
    for (int i = 0; i < latent_dim; ++i) {
        float epsilon = static_cast<float>(rand()) / RAND_MAX; // Random noise
        latent_sample[i] = latent_mean[i] + std::exp(0.5f * latent_log_var[i]) * epsilon;
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    float input[input_dim] = {1.0f, 0.5f, -0.3f, 0.8f, 0.1f, 0.0f, -0.2f, 0.4f}; // Example input
    float latent_mean[latent_dim] = {0};
    float latent_log_var[latent_dim] = {0};
    float latent_sample[latent_dim] = {0};
    float output[output_dim] = {0};

    // Forward pass
    encoder(input, latent_mean, latent_log_var);
    sample_latent(latent_mean, latent_log_var, latent_sample);
    decoder(latent_sample, output);

    // Output results
    std::cout << "Reconstructed output: ";
    for (int i = 0; i < output_dim; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
