import numpy as np
from perceptron import Perceptron
import data as d

class MLP():

    # initialization
    def __init__(self, inputs, hidden_layer_perceptrons, output_neurons):
        self.perceptrons = []
        # add hidden layer
        self.perceptrons.append([Perceptron(inputs) for x in range(hidden_layer_perceptrons)])
        # add output layer
        self.perceptrons.append([Perceptron(hidden_layer_perceptrons) for x in range(output_neurons)])

    # forward step
    def forward_step(self):
        self.outputs = []
        self.outputs.append(inputs)

        # put the input in one layer and compute the output, which is the input for the next layer
        for l in range(len(self.perceptrons)):
            outputs = np.array([p.forward_step(self.outputs[-1]) for p in self.perceptrons[layer]])
            self.outputs.append(outputs)

    # backpropagation step
    def backprop_step(self):
        pass

# initialize MLP, load data, train and visualize
mlp = MLP(inputs=2, hidden_layer_perceptrons=4, output_neurons=1)
data = d.load_data_and()




# main
def main():
    perceptron = perc.Perceptron(10)

if __name__ == '__main__':
    main()
