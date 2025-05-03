#include "NeuralNetwork.h"

int main() {
    CSVReader reader("data.csv", {"feature1", "feature2", "feature3"}, {"label1", "label2"});

    SGD sgd;
    NeuralNetwork net(0.01);
    net.addLayer(3, 10, &sgd);
    net.addLayer(10, 10, &sgd);
    net.addLayer(10, 10, &sgd);
    net.addLayer(10, 10, &sgd);
    net.addLayer(10, 2, &sgd); 

    net.trainEpochs(reader.inputs, reader.targets, 1000, 100);

    
    net.printPredictions(reader.inputs);
}
