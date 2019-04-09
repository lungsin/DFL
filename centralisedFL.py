from mpi4py import MPI
import os
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

SERVER = 0 # rank 0 is the server

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
    x_test = split_by_rank(x_test)
    y_test = split_by_rank(y_test)
    
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
    
    model.add(Convolution2D(32, kernel_size = (3, 3), activation='relu', input_shape=(28, 28, 1)))
    
    model.add(Convolution2D(32, kernel_size = (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))
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
            print("Server's Epoch: " + str(e + 1))
        # broadcast server's model
        weights = model.get_weights()
        weights = comm.bcast(weights, root=0)   
        model.set_weights(weights)
        
        model.fit(x_train, y_train, batch_size=32, nb_epoch=1, verbose=1)
        
        # federate step
        data = [x_train.shape[0], model.get_weights()]
        data = comm.gather(data, root=0)
        if rank == 0:
            weights = [np.zeros(l.shape) for l in data[0][1]]
            
            for i in range(len(weights)):
                N = 0.0
                for n, w in data:
                    weights[i] += n * w[i]
                    N += n
                
                weights[i] /= N
                 
            model.set_weights(weights)
            	
    score = model.evaluate(x_test, y_test, verbose=0)
        
        
main()