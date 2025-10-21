from mpi4py import MPI
import sys

def print_hello(rank, size, name):
  msg = "Hola Mundo! Soy el proceso {0} de {1} sobre {2}.\n"
  sys.stdout.write(msg.format(rank, size, name))

if __name__ == "__main__":
  size = MPI.COMM_WORLD.Get_size() # Obtiene el numero de procesos
  rank = MPI.COMM_WORLD.Get_rank() # Obtiene el identificador del proceso
  name = MPI.Get_processor_name() # Obtiene el nombre de la máquina
  print_hello(rank, size, name)