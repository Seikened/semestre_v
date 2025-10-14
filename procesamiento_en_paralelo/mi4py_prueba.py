# pi_montecarlo_mpi.py
from mpi4py import MPI
import numpy as np
import sys

def split_evenly(N_total: int, size: int, rank: int) -> int:
    """Reparte N_total lo más parejo posible entre 'size' procesos y devuelve la cuota de 'rank'."""
    base = N_total // size
    resto = N_total % size
    # Los primeros 'resto' ranks reciben un elemento extra
    return base + (1 if rank < resto else 0)

def simulate_hits(N_local: int, seed: int) -> int:
    """Cuenta puntos dentro del círculo unitario usando N_local samples."""
    rng = np.random.default_rng(seed)
    # Generamos pares (x, y) en [-1, 1] x [-1, 1]
    x = rng.uniform(-1.0, 1.0, N_local)
    y = rng.uniform(-1.0, 1.0, N_local)
    # Dentro del círculo: x^2 + y^2 <= 1
    return int(np.count_nonzero(x * x + y * y <= 1.0))

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Desde CLI puedes pasar N_total; por defecto usa un tamaño decente
    if rank == 0:
        if len(sys.argv) > 1:
            try:
                N_total = int(sys.argv[1])
            except ValueError:
                raise SystemExit("Usage: mpiexec -n K python pi_montecarlo_mpi.py [N_total:int]")
        else:
            N_total = 10_000_000  # 10 millones por defecto
    else:
        N_total = None

    # Difundimos N_total a todos
    N_total = comm.bcast(N_total, root=0)

    # Cada rank calcula su cuota de trabajo
    N_local = split_evenly(N_total, size, rank)

    # Simulación local (semilla distinta por rank para reproducibilidad)
    hits_local = simulate_hits(N_local, seed=42 + rank)

    # Sumamos los aciertos de todos los procesos en root
    hits_total = comm.reduce(hits_local, op=MPI.SUM, root=0)

    if rank == 0:
        pi_est = 4.0 * hits_total / float(N_total)
        print(f"Estimación de π = {pi_est:.6f}  |  N_total = {N_total}  |  ranks = {size}")

if __name__ == "__main__":
    main()