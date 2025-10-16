from mpi4py import MPI
import numpy as np
from collections import Counter

DTYPE = np.float64



def parametros():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #print(f"Soy el proceso {rank} de un total de {size} procesos")
    return comm, rank, size

def gather_comunicacion():
    comm, rank, size = parametros()
    
    if size != 4:
        if rank == 0:
            print("Este programa debe ejecutarse con 4 procesos")
        comm.Abort()

    data = (rank + 1)**2
    gathered_data = comm.gather(data, root=0)

    if rank == 0:
        print(f"Rank {rank} ... recibiendo datos")
        for i in range(size):
            valor = gathered_data[i]
            print(f"[{i}] Rank {rank} ... procesando el valor: {valor}")

def main_matriz():
    comm, rank, size = parametros()
    
    send_data = np.arange(size, dtype=DTYPE) * (rank + 1)
    receive_data = np.empty(size, dtype=DTYPE)
    comm.Alltoall(send_data,receive_data)
    print(f"Rank {rank} ... enviando {send_data} y recibiendo {receive_data}")

def reduccion():
    comm, rank, size = parametros()
    
    
    data_size = 4
    send_data = np.arange(data_size, dtype=DTYPE) * (rank + 1)
    receive_data = np.empty(data_size, dtype=DTYPE)

    print(f"Rank {rank} ... enviando {send_data}")
    comm.Reduce(send_data, receive_data, op=MPI.SUM, root=0)
    print(f" Despues de la reducción Rank {rank} ... enviando {send_data} y recibiendo {receive_data}")



def tokenize(text):
    # minimal tokenizer; mejora si quieres normalizar: lower(), quitar signos, etc.
    return [w for w in text.lower().split() if w]

def chunk_bounds(n_items, size, rank):
    # split proportional: ranks 0..(n_extra-1) get one extra
    base = n_items // size
    n_extra = n_items % size
    start = rank * base + min(rank, n_extra)
    stop  = start + base + (1 if rank < n_extra else 0)
    return start, stop

def distribute_text(comm, rank, size, raw_text):
    if rank == 0:
        tokens = tokenize(raw_text)
        n = len(tokens)
        slices = []
        for r in range(size):
            a, b = chunk_bounds(n, size, r)
            slices.append(tokens[a:b])
    else:
        slices = None
    # scatter Python objects; mpi4py will pickle/unpickle
    my_tokens = comm.scatter(slices, root=0)
    return my_tokens

def local_count(tokens_chunk):
    return Counter(tokens_chunk)

def merge_counts(comm, local_counter, everyone_needs_result=False):
    if everyone_needs_result:
        return comm.allreduce(local_counter, op=MPI.SUM)
    else:
        return comm.reduce(local_counter, op=MPI.SUM, root=0)

def parallel_wordcount(raw_text=None, everyone_needs_result=False):
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    # Root may read the big text from disk; others pass None
    # Si no quieres broadcast del texto completo, haz scatter de tokens como abajo.
    my_tokens = distribute_text(comm, rank, size, raw_text if rank == 0 else None)
    local_ctr = local_count(my_tokens)
    global_ctr = merge_counts(comm, local_ctr, everyone_needs_result=everyone_needs_result)

    if (not everyone_needs_result and rank == 0) or (everyone_needs_result):
        # Only root prints in Reduce; all can print in Allreduce
        most_common = global_ctr.most_common(10)
        print(f"[rank {rank}] top-10:", most_common)

    return global_ctr if ((everyone_needs_result) or (rank == 0)) else None




def scan_metodo():
    comm, rank, size = parametros()
    
    data_size = 4
    send_data = np.arange(data_size, dtype=DTYPE) * (rank + 1)
    receive_data = np.empty(data_size, dtype=DTYPE)
    comm.Scan(send_data, receive_data, op=MPI.SUM)
    print(f" Tarea y la suma acumulada Rank {rank} ... enviando {send_data} y recibiendo {receive_data[1]}")



if __name__ =="__main__":
    #parallel_wordcount("Este es un texto de ejemplo para contar  este palabras en paralelo.")
    scan_metodo()