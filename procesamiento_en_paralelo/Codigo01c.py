import concurrent.futures
import time
import random
import os

def estudiante_task(id_estudiante):
    """
    Tarea de un estudiante - ahora es pura (sin side effects)
    """
    # Cada estudiante tarda un tiempo aleatorio
    tiempo_tarea = random.uniform(0.1, 0.5)
    time.sleep(tiempo_tarea)
    
    # Calificación aleatoria
    calificacion = random.randint(60, 100)
    
    # Devolver resultado en lugar de modificar recurso compartido
    return {
        'id': id_estudiante,
        'tiempo': tiempo_tarea,
        'calificacion': calificacion,
        'mensaje': f"✅ Estudiante {id_estudiante} terminó - Calificación: {calificacion}/100"
    }

def profesor_monitor(futures, total_estudiantes):
    """
    Monitorea el progreso de las tareas
    """
    progreso_anterior = -1
    
    while True:
        # Contar tareas completadas
        completadas = sum(1 for f in futures if f.done())
        
        # Mostrar progreso si cambió
        if completadas != progreso_anterior:
            print(f"📊 Profesor: {completadas}/{total_estudiantes} estudiantes terminaron")
            progreso_anterior = completadas
        
        # Verificar si todos terminaron
        if completadas >= total_estudiantes:
            break
        
        # Esperar antes de revisar nuevamente
        time.sleep(0.3)

def main_concurrent_futures():
    print("=" * 60)
    print("🏫 SIMULACIÓN CON concurrent.futures")
    print("=" * 60)
    print("🎯 Usando ThreadPoolExecutor para paralelismo")
    print("=" * 60)
    
    num_estudiantes = 12
    resultados = []
    
    print(f"\n🎯 Creando {num_estudiantes} tareas de estudiantes...")
    
    # Usar ThreadPoolExecutor para manejar los hilos
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=4,  # Número máximo de hilos simultáneos
        thread_name_prefix="Estudiante"
    ) as executor:
        
        # Enviar todas las tareas al executor
        futures = [
            executor.submit(estudiante_task, i)
            for i in range(num_estudiantes)
        ]
        
        print("🚀 Tareas enviadas al ThreadPoolExecutor...")
        
        # Iniciar monitoreo en el hilo principal
        profesor_monitor(futures, num_estudiantes)
        
        # Recolectar resultados
        print("\n⏳ Recolectando resultados...")
        for future in concurrent.futures.as_completed(futures):
            try:
                resultado = future.result(timeout=1.0)
                resultados.append(resultado)
                print(resultado['mensaje'])
            except concurrent.futures.TimeoutError:
                print("⏰ Timeout en una tarea")
            except Exception as e:
                print(f"❌ Error en tarea: {e}")
    
    # Procesar resultados finales
    print("\n" + "=" * 60)
    print("🏆 RESULTADOS FINALES - concurrent.futures")
    print("=" * 60)
    
    calificaciones = [r['calificacion'] for r in resultados]
    promedios_tiempo = sum(r['tiempo'] for r in resultados) / len(resultados)
    
    print(f"📈 Calificación promedio: {sum(calificaciones)/len(calificaciones):.1f}/100")
    print(f"⏱️  Tiempo promedio por tarea: {promedios_tiempo:.2f}s")
    print(f"🎯 Mejor calificación: {max(calificaciones)}/100")
    print(f"📉 Peor calificación: {min(calificaciones)}/100")
    
    # Información del executor
    print(f"\n🔧 Máximo de workers: 4")
    print(f"📊 Tareas completadas: {len(resultados)}")
    
    return resultados