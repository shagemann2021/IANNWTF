import numpy as np
import util as u

class Perceptron():

    # constructor
    def __init__(self, input_units:int, bias=np.random.randn(), alpha=1):
        weights = []
        for x in range(input_units + 1):
            weights.append(np.random.randn())
        self.weights = weights
        self.bias = bias
        self.alpha = alpha

    # forward step function
    # calculates the activation of the perceptron
    def forward_step(self, inputs):
        self.raw = np.concatenate((inputs, np.array([self.bias])))
        self.sum_inputs = np.sum(self.raw @ self.weights)
        self.out = u.sigmoid(self.sum_inputs)
        return self.out

    # function to update the parameters
    def update(self, delta):
        self.weights -= self.alpha*delta*self.raw

    def get_output(self):
        """Get the output after the activation function."""
        return self.out

    def get_input(self):
        """Get the input to the perceptron before the activation function."""
        return self.sum_inputs

    def get_weights(self):
        return self.weights
