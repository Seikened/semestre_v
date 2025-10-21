#hello.py
from mpi4py import MPI

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print ("Hola mundo desde el proceso ", rank)