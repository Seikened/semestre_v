import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

#Ejercicio 1
def tarea(id):
    print(f"Tarea {id} iniciada")
    time.sleep(2)
    print(f"Tarea {id} finalizada")

def ejercicio_1():
    # Creación de hilos
    hilos = []
    for i in range(3):
        hilo = threading.Thread(target=tarea, args=(i,))
        hilos.append(hilo)
        hilo.start()

    # Esperar a que todos terminen
    for hilo in hilos:
        hilo.join()

    print("Todas las tareas han terminado")

#Ejercicio 2
def tarea_automática(id):
    print(f"Tarea {id} iniciada")
    time.sleep(2)
    return f"Tarea {id} completada"

def ejercicio_2():
    # Paralelización automática con 3 workers
    with ThreadPoolExecutor(max_workers=3) as executor:
        futuros = [executor.submit(tarea_automática, i) for i in range(3)]
        resultados = [futuro.result() for futuro in futuros]

    print(resultados)

#Ejercicio 3
def procesar_chunk(chunk):
    return sum(chunk)

def ejercicio_3():
    datos = list(range(100))
    num_procesos = 4
    chunk_size = len(datos) // num_procesos
    chunks = [datos[i:i+chunk_size] for i in range(0, len(datos), chunk_size)]

    with mp.Pool(processes=num_procesos) as pool:
        resultados = pool.map(procesar_chunk, chunks)

    total = sum(resultados)
    print(f"Suma total: {total}")

#Ejercicio 4
def productor(q, id):
    q.put(f"Mensaje del productor {id}")

def consumidor(q):
    while not q.empty():
        mensaje = q.get()
        print(f"Consumiendo: {mensaje}")

def ejercicio_4():
    q = mp.Queue()
    procesos = []
    # Productores
    for i in range(3):
        p = mp.Process(target=productor, args=(q, i))
        procesos.append(p)
        p.start()
    for p in procesos:
        p.join()
    # Consumidor
    c = mp.Process(target=consumidor, args=(q,))
    c.start()
    c.join()

#Ejercicio 5
contador = 0
lock = threading.Lock()

def incrementar():
     global contador
     for _ in range(100000):
         with lock:
             contador += 1

def ejercicio_5():
    hilos = []
    for _ in range(4):
        hilo = threading.Thread(target=incrementar)
        hilos.append(hilo)
        hilo.start()

    for hilo in hilos:
        hilo.join()

    print(f"Contador final: {contador}")

#Ejercicio 6
semáforo = threading.Semaphore(2)  # Máximo 2 hilos simultáneos

def recurso_limitado(id):
    with semáforo:
        print(f"Hilo {id} accediendo al recurso")
        time.sleep(2)
    print(f"Hilo {id} liberó el recurso")

def ejercicio_6():
    hilos = []
    for i in range(5):
        hilo = threading.Thread(target=recurso_limitado, args=(i,))
        hilos.append(hilo)
        hilo.start()

    for hilo in hilos:
        hilo.join()

#Ejercicio 7
barrera = threading.Barrier(3)

def tarea_barrera(id):
    print(f"Hilo {id} llegó a la barrera")
    barrera.wait()
    print(f"Hilo {id} pasó la barrera")

def ejercicio_7():
    hilos = []
    for i in range(3):
        hilo = threading.Thread(target=tarea_barrera, args=(i,))
        hilos.append(hilo)
        hilo.start()

    for hilo in hilos:
        hilo.join()

#Ejercicio 8
evento = threading.Event()

def esperar_evento(id):
    print(f"Hilo {id} esperando el evento")
    evento.wait()
    print(f"Hilo {id} recibió el evento")

def disparar_evento():
    import time
    time.sleep(2)
    evento.set()
    print("Evento disparado")

def ejercicio_8():
    hilo_espera = threading.Thread(target=esperar_evento, args=(1,))
    hilo_disparo = threading.Thread(target=disparar_evento)

    hilo_espera.start()
    hilo_disparo.start()

    hilo_espera.join()
    hilo_disparo.join()

#Ejercicio 9
def factorial(n):
    resultado = 1
    for i in range(1, n+1):
        resultado *= i
    return resultado

def ejercicio_9():
    números = [1000, 1000, 1000, 1000]
    inicio = time.time()
    with mp.Pool(processes=4) as pool:
        resultados = pool.map(factorial, números)
    fin = time.time()
    print(f"Tiempo paralelo: {fin-inicio} segundos")

#Ejercicio 10
def cuadrado(n):
    time.sleep(1)
    return n * n

def ejercicio_10():
    with ProcessPoolExecutor() as executor:
        futuros = [executor.submit(cuadrado, i) for i in range(5)]
        resultados = [f.result() for f in futuros]
    print(resultados)

if __name__ == "__main__":
    # Ejecutar ejercicio específico
    print("=== EJERCICIO 1 ===")
    ejercicio_1()

    # Ejecutar ejercicio específico
    print("=== EJERCICIO 2 ===")
    ejercicio_2()

    # Ejecutar ejercicio específico
    print("=== EJERCICIO 3 ===")
    ejercicio_3()

    # Ejecutar ejercicio específico
    print("=== EJERCICIO 4===")
    ejercicio_4()

    # Ejecutar ejercicio específico
    print("=== EJERCICIO 5 ===")
    ejercicio_5()

    # Ejecutar ejercicio específico
    print("=== EJERCICIO 6 ===")
    ejercicio_6()
    
    # Ejecutar ejercicio específico
    print("=== EJERCICIO 7 ===")
    ejercicio_7()

    # Ejecutar ejercicio específico
    print("=== EJERCICIO 8 ===")
    ejercicio_8()

    # Ejecutar ejercicio específico
    print("=== EJERCICIO 9 ===")
    ejercicio_9()

    # Ejecutar ejercicio específico
    print("=== EJERCICIO 10 ===")
    ejercicio_10()