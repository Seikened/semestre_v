from mpi4py import MPI
import numpy as np
import random

def monte_carlo_pi(num_samples):
    count = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x**2 + y**2 <= 1:
            count += 1
    return count

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define el número de muestras que generará cada proceso. Ajustar como necesite
    num_samples_per_proc = 1000000  
    
    # Calcula el número total de muestras multiplicando el número de muestras por proceso por el número total de procesos.
    total_samples = num_samples_per_proc * size

    # Cada proceso ejecuta su propia simulación de Monte Carlo y almacena el resultado en local_count   
    local_count = monte_carlo_pi(num_samples_per_proc)

    # Recopilar todos los recuentos locales en el proceso raíz y los suma utilizando la operación de reducción.
    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

    # Comprueba si el proceso actual es el proceso raíz, utilizando el recuento total de puntos
    # dentro del cuarto de círculo y el número total de muestras
    if rank == 0:
        # El proceso raíz calcula la estimación final de Pi
        pi_estimate = (4.0 * total_count) / total_samples
        print(f"Estimated Pi: {pi_estimate}")

if __name__ == "__main__":
    main()