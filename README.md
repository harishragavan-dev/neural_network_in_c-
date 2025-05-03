# Neural Network C++ Library

A lightweight C++ implementation of a neural network with support for:

- Dense (fully connected) layers
- Activation functions (currently Sigmoid)
- Optimizers (SGD and Adam)
- Training from CSV input
- Prediction output

## Features

- **Layer Stacking**: Easily add multiple dense layers.
- **Optimizers**: Plug in SGD or Adam for weight updates.
- **Data Loading**: Read input and target data from CSV files.
- **Training Loop**: Train the model over multiple epochs with loss reporting.
- **Prediction Interface**: Make predictions on input data after training.

## Getting Started

### 1. Build Your Network

To use the library, include `NeuralNetworkLib.h` and build the neural network. Here’s an example:

```cpp
#include "NeuralNetworkLib.h"

int main() {
    // Load data from CSV
    CSVReader reader("data.csv", {"feature1", "feature2", "feature3"}, {"label1", "label2"});

    // Choose optimizer
    SGD sgd;

    // Initialize neural network with a learning rate
    NeuralNetwork net(0.01); // learning rate

    // Add layers to the network
    net.addLayer(3, 10, &sgd);
    net.addLayer(10, 10, &sgd);
    net.addLayer(10, 2, &sgd);

    // Train the network
    net.trainEpochs(reader.inputs, reader.targets, 1000, 100); // train for 1000 epochs, report every 100

    // Make predictions on training data
    net.printPredictions(reader.inputs); 
}
```

