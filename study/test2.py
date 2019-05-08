#coding:utf-8

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mseLoss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean(dtype=float)

'''
- 2 input 
- a hidden layer with 2 neurons (h1, h2)
- an output layer with 1 neuron (o1)
'''

class Neuron:
    
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, input):
        total = np.dot(self.weights, input) + self.bias
        return sigmoid(total)

class OurNeuralNetwork:

    def __init__(self):
        #leanrn_rate
        self.leanrn_rate = 0.1

       # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

       # bias
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

        return o1

    def train(self, data, all_y_trues, epoches):
        '''
        - data is a (nx2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        - epoches, number of times to loop though the entire dataset

        - NeuralNetwork:

        input  hidden_layer  output

                    w1
            weight ---- h1 
                 w3 \ /    \w5
                     x      >----o1 gender
                 w2 / \    /w6
            height ---- h2
                    w4

        '''

        for  epoch in range(epoches):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 repesents "partial L / partial w1"

                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                # y_pred = sigmoid(self.w5 * h1 + self.w6 * h2 + b3)
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                # sum_h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1  = deriv_sigmoid(sum_h1) 

                # Neuron h2
                # sum_h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2) 

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= self.leanrn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= self.leanrn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= self.leanrn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= self.leanrn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= self.leanrn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= self.leanrn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= self.leanrn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= self.leanrn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= self.leanrn_rate * d_L_d_ypred * d_ypred_d_b3

                # --- Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mseLoss(all_y_trues, y_preds)
                    print ('Epoch: %d loss: %.5f'%(epoch, loss))
                pass

def main():
    data = np.array([
        [-2, -1],
        [25, 6],
        [17, 4],
        [-15, -6],
    ])

    all_y_trues = np.array([
        1,
        0,
        0,
        1
    ])

    # start train out network!
    network = OurNeuralNetwork()
    network.train(data, all_y_trues, 100000)
    pass

if __name__ == '__main__':
    main()