import multiprocessing
import time
import random
import os

def estudiante_process(id_estudiante, result_queue):
    """
    Proceso de un estudiante - ejecutado en proceso separado
    """
    # Información del proceso
    pid = os.getpid()
    
    # Cada estudiante tarda un tiempo aleatorio
    tiempo_tarea = random.uniform(0.1, 0.5)
    time.sleep(tiempo_tarea)
    
    # Calificación aleatoria
    calificacion = random.randint(60, 100)
    
    # Mensaje con info del proceso
    mensaje = f"✅ Estudiante {id_estudiante} (PID: {pid}) - Calificación: {calificacion}/100"
    print(mensaje)
    
    # Enviar resultado a través de la cola
    result_queue.put({
        'id': id_estudiante,
        'pid': pid,
        'tiempo': tiempo_tarea,
        'calificacion': calificacion,
        'mensaje': mensaje
    })

def profesor_monitor_process(result_queue, total_estudiantes, processes):
    """
    Monitorea el progreso de los procesos
    """
    resultados_recibidos = 0
    progreso_anterior = -1
    
    while resultados_recibidos < total_estudiantes:
        # Intentar obtener resultado sin bloquear
        try:
            resultado = result_queue.get(timeout=0.1)
            resultados_recibidos += 1
        except:
            # No hay resultados disponibles aún
            pass
        
        # Mostrar progreso si cambió
        if resultados_recibidos != progreso_anterior:
            activos = sum(1 for p in processes if p.is_alive())
            print(f"📊 Profesor: {resultados_recibidos}/{total_estudiantes} terminaron | Procesos activos: {activos}")
            progreso_anterior = resultados_recibidos
        
        time.sleep(0.3)

def main_multiprocessing():
    print("=" * 60)
    print("🏫 SIMULACIÓN CON multiprocessing")
    print("=" * 60)
    print("🎯 Usando procesos separados (real paralelismo)")
    print("=" * 60)
    
    num_estudiantes = 12
    resultados = []
    
    # Cola para comunicación entre procesos
    result_queue = multiprocessing.Queue()
    
    # Lista de procesos
    processes = []
    
    print(f"\n🎯 Creando {num_estudiantes} procesos de estudiantes...")
    print(f"🔧 Núcleos disponibles: {multiprocessing.cpu_count()}")
    
    # Crear procesos
    for i in range(num_estudiantes):
        process = multiprocessing.Process(
            target=estudiante_process,
            args=(i, result_queue),
            name=f"Estudiante-{i}"
        )
        processes.append(process)
    
    # Iniciar procesos
    print("🚀 Iniciando procesos...")
    start_time = time.time()
    
    for process in processes:
        process.start()
    
    # Monitorear progreso
    profesor_monitor_process(result_queue, num_estudiantes, processes)
    
    # Esperar a que todos los procesos terminen
    for process in processes:
        process.join()
    
    end_time = time.time()
    tiempo_total = end_time - start_time
    
    # Recolectar todos los resultados
    print("\n⏳ Recolectando resultados finales...")
    while not result_queue.empty():
        try:
            resultado = result_queue.get_nowait()
            resultados.append(resultado)
        except:
            break
    
    # Procesar resultados finales
    print("\n" + "=" * 60)
    print("🏆 RESULTADOS FINALES - multiprocessing")
    print("=" * 60)
    
    calificaciones = [r['calificacion'] for r in resultados]
    pids_unicos = len(set(r['pid'] for r in resultados))
    promedios_tiempo = sum(r['tiempo'] for r in resultados) / len(resultados)
    
    print(f"📈 Calificación promedio: {sum(calificaciones)/len(calificaciones):.1f}/100")
    print(f"⏱️  Tiempo promedio por tarea: {promedios_tiempo:.2f}s")
    print(f"⏱️  Tiempo total de ejecución: {tiempo_total:.2f}s")
    print(f"🎯 Mejor calificación: {max(calificaciones)}/100")
    print(f"📉 Peor calificación: {min(calificaciones)}/100")
    print(f"🔧 Procesos únicos utilizados: {pids_unicos}")
    
    # Mostrar información de procesos
    print(f"\n🖥️  Núcleos CPU: {multiprocessing.cpu_count()}")
    print(f"📊 Procesos creados: {len(processes)}")
    
    return resultados

if __name__ == "__main__":
    # Ejecutar la simulación principal
    main_multiprocessing()