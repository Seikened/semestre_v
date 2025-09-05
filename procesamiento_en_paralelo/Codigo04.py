import threading
import time
import numpy as np
from queue import Queue

def calcular_pi_threading(num_puntos, num_threads=4):
    """
    Calcula π usando el método Monte Carlo con threading
    """
    puntos_por_thread = num_puntos // num_threads
    puntos_circulo = [0] * num_threads
    threads = []
    
    def worker(thread_id, puntos):
        circulo_local = 0
        np.random.seed(thread_id + int(time.time()))
        for _ in range(puntos):
            x, y = np.random.random(), np.random.random()
            if x**2 + y**2 <= 1:
                circulo_local += 1
        puntos_circulo[thread_id] = circulo_local
    
    # Tiempo secuencial de referencia
    start_seq = time.time()
    worker(0, num_puntos)
    tiempo_secuencial = time.time() - start_seq
    
    # Ejecución paralela
    start_par = time.time()
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i, puntos_por_thread))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    tiempo_paralelo = time.time() - start_par
    
    # Calcular π y métricas
    total_circulo = sum(puntos_circulo)
    pi_estimado = 4 * total_circulo / num_puntos
    
    # Medidas de rendimiento
    aceleracion = tiempo_secuencial / tiempo_paralelo
    eficiencia = aceleracion / num_threads
    
    return {
        'pi_estimado': pi_estimado,
        'tiempo_secuencial': tiempo_secuencial,
        'tiempo_paralelo': tiempo_paralelo,
        'aceleracion': aceleracion,
        'eficiencia': eficiencia,
        'num_threads': num_threads
    }

# Ejecutar y mostrar resultados
if __name__ == "__main__":
    resultados = calcular_pi_threading(10_000_000, num_threads=4)
    
    print("=== RESULTADOS CON THREADING ===")
    print(f"π estimado: {resultados['pi_estimado']}")
    print(f"Tiempo secuencial: {resultados['tiempo_secuencial']:.4f} s")
    print(f"Tiempo paralelo: {resultados['tiempo_paralelo']:.4f} s")
    print(f"Aceleración (S): {resultados['aceleracion']:.2f}")
    print(f"Eficiencia (E): {resultados['eficiencia']:.2%}")
    print(f"Número de threads: {resultados['num_threads']}")