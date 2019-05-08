import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mseLoss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class Neuron:
    
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, input):
        total = np.dot(self.weights, input) + self.bias
        return sigmoid(total)

class OurNeuralNetwork:

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        input_o1 = np.array([out_h1, out_h2])
        out_o1 = self.o1.feedforward(input_o1)

        return out_o1

def main():
    weights = np.array([0, 1])  # w_1 = 0, w_2 = 1
    bias = 4                    # b = 4
    n = Neuron(weights, bias)

    x = np.array([2, 3])        # x1 = 2, x2 = 3

    print (n.feedforward(x))

    network = OurNeuralNetwork()
    print (network.feedforward(x))
    pass

if __name__ == '__main__':
    main()