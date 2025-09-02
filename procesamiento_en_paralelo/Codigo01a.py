import threading
import time
import random

def estudiante_thread(id_estudiante, resultados, lock):
    """Versión modificada para retornar resultados"""
    tiempo_tarea = random.uniform(0.1, 0.5)
    
    with lock:
        print(f"🎓 Estudiante {id_estudiante} comenzó su tarea")
    
    time.sleep(tiempo_tarea)
    
    calificacion = random.randint(60, 100)
    resultado = {
        'id': id_estudiante,
        'calificacion': calificacion,
        'tiempo': tiempo_tarea
    }
    
    with lock:
        resultados[id_estudiante] = resultado
        print(f"✅ Estudiante {id_estudiante} terminó - Calificación: {calificacion}/100")
    
    return resultado

def main_threading():
    """Implementación original modificada para retornar resultados"""
    lock = threading.Lock()
    resultados = {}
    num_estudiantes = 8
    
    # Crear hilos
    threads = []
    for i in range(num_estudiantes):
        thread = threading.Thread(
            target=estudiante_thread,
            args=(i, resultados, lock),
            name=f"Estudiante-{i}"
        )
        threads.append(thread)
    
    # Ejecutar
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Convertir a lista ordenada
    return [resultados[i] for i in range(num_estudiantes)]