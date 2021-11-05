import numpy as np
import util as u

class Perceptron():

    # constructor
    def __init__(self, input_units:int, bias=np.random.randn(), alpha=1):
        weights = []
        for x in range(input_units):
            weights.append(np.random.randn())
        self.weights = weights
        self.bias = bias
        self.alpha = alpha

    # forward step function
    # calculates the activation of the perceptron
    def forward_step(self, inputs):
        self.raw = np.concatenate((inputs, np.array([1])))
        self.sum_inputs = np.sum(self.raw @ self.weights)
        self.out = u.sigmoid(self.sum_inputs)
        return self.out

    # function to update the parameters
    def update(self, delta):
        self.weights -= alpha*delta*self.raw
