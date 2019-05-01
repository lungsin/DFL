from mpi4py import MPI
import os
import sys
import graph

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# W: mixing matrix for network topology
W = graph.ring_graph(rank, size)

import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

def split_by_rank(arr): # each worker have a chunk of data
    return np.split(arr, size)[rank]

def splited_worker_data():
    # Load pre-shuffled MNIST data into train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # each worker can only see some part of the data
    x_train = split_by_rank(x_train)
    y_train = split_by_rank(y_train)
    
    print(str(rank) + " : " + str(x_train.shape))
    
    # reshape to (img_cols, img_rows, img_depth)
    img_cols = x_train.shape[1]
    img_rows = x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, 1)
    x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def build_model():
    # Build the model 
    model = Sequential()
    
    model.add(Flatten(input_shape=(28, 28, 1)))
    # model.add(Dropout(rate=0.25))
    # model.add(Dense(128))
    # model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
    return model
    
def main():
    (x_train, y_train), (x_test, y_test) = splited_worker_data()
    
    model = build_model()
    
    epoch = 10
    for e in range(epoch):
        if rank == 0:
            print("Rank 0's Epoch: ", e + 1)
            sys.stdout.flush()
            
        model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=0)
        
        # federate step
        data = [rank, model.get_weights()]
        data = comm.allgather(data)
    
        weights = [np.zeros(l.shape) for l in data[0][1]]
        
        for i in range(len(weights)):
            # N = 0.0
            
            for id_neighbour, w in data:
                weights[i] += W[id_neighbour] * w[i]
                # N += n
            
            # weights[i] /= N
                
        model.set_weights(weights)
    
    # after training is over, try to test
    if rank == 0:            	
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
main()