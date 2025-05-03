#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>

// === Activation ===
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// === Optimizers ===
class Optimizer {
public:
    virtual void update(std::vector<std::vector<double>>& weights,
                        std::vector<double>& biases,
                        const std::vector<std::vector<double>>& dW,
                        const std::vector<double>& dB,
                        double lr) = 0;
};

class SGD : public Optimizer {
public:
    void update(std::vector<std::vector<double>>& weights,
                std::vector<double>& biases,
                const std::vector<std::vector<double>>& dW,
                const std::vector<double>& dB,
                double lr) override {
        for (size_t i = 0; i < weights.size(); ++i)
            for (size_t j = 0; j < weights[0].size(); ++j)
                weights[i][j] -= lr * dW[i][j];
        for (size_t i = 0; i < biases.size(); ++i)
            biases[i] -= lr * dB[i];
    }
};

class Adam : public Optimizer {
public:
    Adam() : t(0), beta1(0.9), beta2(0.999), epsilon(1e-8) {}

    void update(std::vector<std::vector<double>>& weights,
                std::vector<double>& biases,
                const std::vector<std::vector<double>>& dW,
                const std::vector<double>& dB,
                double lr) override {
        ++t;

        if (mW.empty()) {
            mW = dW; vW = dW;
            mB = dB; vB = dB;
            for (auto& row : vW) for (auto& val : row) val = 0.0;
            for (auto& row : mW) for (auto& val : row) val = 0.0;
            for (auto& val : vB) val = 0.0;
            for (auto& val : mB) val = 0.0;
        }

        for (size_t i = 0; i < dW.size(); ++i) {
            for (size_t j = 0; j < dW[0].size(); ++j) {
                mW[i][j] = beta1 * mW[i][j] + (1 - beta1) * dW[i][j];
                vW[i][j] = beta2 * vW[i][j] + (1 - beta2) * dW[i][j] * dW[i][j];
                double mW_hat = mW[i][j] / (1 - pow(beta1, t));
                double vW_hat = vW[i][j] / (1 - pow(beta2, t));
                weights[i][j] -= lr * mW_hat / (sqrt(vW_hat) + epsilon);
            }
            mB[i] = beta1 * mB[i] + (1 - beta1) * dB[i];
            vB[i] = beta2 * vB[i] + (1 - beta2) * dB[i] * dB[i];
            double mB_hat = mB[i] / (1 - pow(beta1, t));
            double vB_hat = vB[i] / (1 - pow(beta2, t));
            biases[i] -= lr * mB_hat / (sqrt(vB_hat) + epsilon);
        }
    }

private:
    double beta1, beta2, epsilon;
    int t;
    std::vector<std::vector<double>> mW, vW;
    std::vector<double> mB, vB;
};

// === Dense Layer ===
class DenseLayer {
public:
    int inputSize, outputSize;
    std::vector<std::vector<double>> weights, dW;
    std::vector<double> biases, dB;
    std::vector<double> input, output, z;
    Optimizer* optimizer;

    DenseLayer(int in, int out, Optimizer* opt) : inputSize(in), outputSize(out), optimizer(opt) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0, 1);

        weights.resize(in, std::vector<double>(out));
        dW.resize(in, std::vector<double>(out));
        for (int i = 0; i < in; ++i)
            for (int j = 0; j < out; ++j)
                weights[i][j] = dist(gen) * sqrt(2.0 / in);

        biases.resize(out, 0.0);
        dB.resize(out, 0.0);
    }

    std::vector<double> forward(const std::vector<double>& x) {
        input = x;
        z.resize(outputSize, 0.0);
        output.resize(outputSize, 0.0);

        for (int j = 0; j < outputSize; ++j) {
            for (int i = 0; i < inputSize; ++i)
                z[j] += x[i] * weights[i][j];
            z[j] += biases[j];
            output[j] = sigmoid(z[j]);
        }
        return output;
    }

    std::vector<double> backward(const std::vector<double>& grad_out, double lr) {
        std::vector<double> grad_in(inputSize, 0.0);
        std::vector<double> delta(outputSize, 0.0);

        for (int j = 0; j < outputSize; ++j)
            delta[j] = grad_out[j] * sigmoid_derivative(z[j]);

        for (int i = 0; i < inputSize; ++i)
            for (int j = 0; j < outputSize; ++j) {
                dW[i][j] = input[i] * delta[j];
                grad_in[i] += weights[i][j] * delta[j];
            }

        for (int j = 0; j < outputSize; ++j)
            dB[j] = delta[j];

        optimizer->update(weights, biases, dW, dB, lr);
        return grad_in;
    }
};

// === Neural Network ===
class NeuralNetwork {
public:
    std::vector<DenseLayer> layers;
    double learningRate;

    NeuralNetwork(double lr) : learningRate(lr) {}

    void addLayer(int input, int output, Optimizer* optimizer) {
        layers.emplace_back(input, output, optimizer);
    }

    std::vector<double> predict(const std::vector<double>& x) {
        std::vector<double> out = x;
        for (auto& layer : layers)
            out = layer.forward(out);
        return out;
    }

    double train(const std::vector<double>& x, const std::vector<double>& y) {
        std::vector<double> pred = predict(x);
        std::vector<double> grad(pred.size());

        double loss = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            loss += 0.5 * pow(pred[i] - y[i], 2);
            grad[i] = pred[i] - y[i];
        }

        for (int i = layers.size() - 1; i >= 0; --i)
            grad = layers[i].backward(grad, learningRate);

        return loss;
    }

    void trainEpochs(const std::vector<std::vector<double>>& inputs,
                     const std::vector<std::vector<double>>& targets,
                     int epochs, int logInterval = 100) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i)
                totalLoss += train(inputs[i], targets[i]);

            if (logInterval > 0 && epoch % logInterval == 0)
                std::cout << "Epoch " << epoch << ", Loss: " << totalLoss << "\n";
        }
    }

    void printPredictions(const std::vector<std::vector<double>>& inputs) {
        for (const auto& input : inputs) {
            std::vector<double> pred = predict(input);
            for (double val : pred)
                std::cout << val << " ";
            std::cout << "\n";
        }
    }
};

// === CSV Reader ===
class CSVReader {
public:
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;

    CSVReader(const std::string& filename,
              const std::vector<std::string>& inputCols,
              const std::vector<std::string>& targetCols) {
        std::ifstream file(filename);
        if (!file.is_open())
            throw std::runtime_error("Failed to open file.");

        std::string line;
        std::getline(file, line);

        std::stringstream ss(line);
        std::string col;
        std::unordered_map<std::string, int> colIndex;
        int idx = 0;
        while (std::getline(ss, col, ',')) {
            colIndex[col] = idx++;
        }

        std::vector<int> inputIdx, targetIdx;
        for (const auto& name : inputCols)
            inputIdx.push_back(colIndex[name]);
        for (const auto& name : targetCols)
            targetIdx.push_back(colIndex[name]);

        while (std::getline(file, line)) {
            std::stringstream lss(line);
            std::string val;
            std::vector<std::string> values;
            while (std::getline(lss, val, ','))
                values.push_back(val);

            std::vector<double> inputRow, targetRow;
            for (int i : inputIdx) inputRow.push_back(std::stod(values[i]));
            for (int i : targetIdx) targetRow.push_back(std::stod(values[i]));

            inputs.push_back(inputRow);
            targets.push_back(targetRow);
        }
    }
};
