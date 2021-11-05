import numpy as np

# data for and
def load_data_and(self):
    data = [[1,0,0],
            [0,1,0],
            [1,1,1],
            [0,0,0]]
    return data

# data for or
def load_data_or(self):
    data = [[1,0,1],
            [0,1,1],
            [1,1,1],
            [0,0,0]]
    return data

# data for not and
def load_data_notand(self):
    data = [[1,0,1],
            [0,1,1],
            [1,1,0],
            [0,0,1]]
    return data

# data for not or
def load_data_notor(self):
    data = [[1,0,0],
            [0,1,0],
            [1,1,0],
            [0,0,1]]
    return data

# data for xor
def load_data_xor(self):
    data = [[1,0,1],
            [0,1,1],
            [1,1,0],
            [0,0,0]]
    return data
