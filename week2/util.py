import numpy as np

# the sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# the sigmoid prime function
def sigmoidprime(x):
    sig = sigmoid(x)
    return sig*(1-sig)
