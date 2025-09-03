from multiprocessing import Pool, Process, current_process, Queue, Lock, Pipe
from typing import Callable


# def proceso():
#     pass


# def worker(num_proces:int=1):
#     with Pool.
#     proceso()


def hello(name, last_name):
    print("hello", name, last_name)



def proceso(funcion: Callable, *args):
    p = Process(target=funcion, args=(*args,))
    return p


if __name__ == "__main__":
    p = proceso(hello, "bob", "smith")
    p.start()
    p.join()
