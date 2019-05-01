import numpy as np 

def ring_graph(id, size):
    arr = np.zeros(size)
    arr[id - 1] = arr[(id + 1) % size] = 0.5
    return arr