import numpy as np

# data for and
def load_data_and(self):
    return np.array([[1,0,0],
                    [0,1,0],
                    [1,1,1],
                    [0,0,0]])

# data for or
def load_data_or(self):
    return np.array([[1,0,1],
                    [0,1,1],
                    [1,1,1],
                    [0,0,0]])

# data for not and
def load_data_notand(self):
    return np.array([[1,0,1],
                    [0,1,1],
                    [1,1,0],
                    [0,0,1]])

# data for not or
def load_data_notor(self):
    return np.array([[1,0,0],
                    [0,1,0],
                    [1,1,0],
                    [0,0,1]])

# data for xor
def load_data_xor(self):
    return np.array([[1,0,1],
                    [0,1,1],
                    [1,1,0],
                    [0,0,0]])
