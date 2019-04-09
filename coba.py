from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = [[1, 2, 3, 4], [1, 2]]
    # data = [1, 2, 3, 4]
else:
    data = None
data = comm.bcast(data, root=0)

if rank:
    print(data)