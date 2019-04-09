# DFL
an implementation of Federated Learning on mnist dataset using MPI. 

## Preparation
1. install mpi implementation in your computer
2. install python dependency
    ```bash
    pip3 install keras tensorflow mpi4py
    ```
    
## Centralised Federated Learning

To run:
```
mpiexec -n <number of nodes> python3 centralisedFL.py
```

To measure the time elapsed:
```
time mpiexec -n <number of nodes> python3 centralisedFL.py
```


## Vanilla MNIST Convolutional Neural Netowork

To run:
```
python3 original.py
```

To measure the time elapsed:
```
time python3 original.py
```

## Experimentation
    The result of the experementation can be seen in the result folder.
