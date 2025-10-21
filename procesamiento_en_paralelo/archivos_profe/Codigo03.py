import threading
import time
import random
from contextlib import contextmanager
from threading import Lock, RLock

# =============================================================================
# 1. INTERBLOQUEO CLÁSICO CON DOS LOCKS
# =============================================================================

def ejemplo_deadlock_clasico():
    print("=== 1. INTERBLOQUEO CLÁSICO ===")
    
    lock_a = Lock()
    lock_b = Lock()
    
    def hilo_1():
        print("Hilo 1: Adquiriendo Lock A...")
        lock_a.acquire()
        print("Hilo 1: Lock A adquirido")
        time.sleep(0.1)  # Simular trabajo
        
        print("Hilo 1: Intentando adquirir Lock B...")
        lock_b.acquire()  # ¡ESPERANDO!
        print("Hilo 1: Lock B adquirido")
        
        # Trabajo con ambos recursos
        print("Hilo 1: Trabajando con ambos locks...")
        time.sleep(0.1)
        
        lock_b.release()
        lock_a.release()
        print("Hilo 1: Locks liberados")
    
    def hilo_2():
        print("Hilo 2: Adquiriendo Lock B...")
        lock_b.acquire()
        print("Hilo 2: Lock B adquirido")
        time.sleep(0.1)  # Simular trabajo
        
        print("Hilo 2: Intentando adquirir Lock A...")
        lock_a.acquire()  # ¡ESPERANDO!
        print("Hilo 2: Lock A adquirido")
        
        # Trabajo con ambos recursos
        print("Hilo 2: Trabajando con ambos locks...")
        time.sleep(0.1)
        
        lock_a.release()
        lock_b.release()
        print("Hilo 2: Locks liberados")
    
    # Crear y ejecutar hilos
    t1 = threading.Thread(target=hilo_1)
    t2 = threading.Thread(target=hilo_2)
    
    t1.start()
    t2.start()
    
    # Esperar un tiempo y verificar si hay interbloqueo
    time.sleep(1)
    print(f"\nEstado después de 1 segundo:")
    print(f"Hilo 1 vivo: {t1.is_alive()}, Hilo 2 vivo: {t2.is_alive()}")
    
    if t1.is_alive() and t2.is_alive():
        print("¡INTERBLOQUEO DETECTADO! Ambos hilos están esperando indefinidamente.")
    
    # Esperar a que terminen (no terminarán debido al deadlock)
    t1.join(timeout=2)
    t2.join(timeout=2)

# =============================================================================
# 2. INTERBLOQUEO CON MÚLTIPLES RECURSOS
# =============================================================================

def ejemplo_deadlock_multiple():
    print("\n=== 2. INTERBLOQUEO CON MÚLTIPLES RECURSOS ===")
    
    locks = [Lock() for _ in range(5)]
    recursos = ["A", "B", "C", "D", "E"]
    
    def trabajador(id, recursos_necesarios):
        print(f"Trabajador {id}: Necesita recursos {recursos_necesarios}")
        
        # Adquirir locks en orden aleatorio (puede causar deadlock)
        adquiridos = []
        try:
            for recurso in recursos_necesarios:
                idx = recursos.index(recurso)
                print(f"Trabajador {id}: Intentando adquirir {recurso}...")
                locks[idx].acquire()
                adquiridos.append(idx)
                print(f"Trabajador {id}: Adquirió {recurso}")
                time.sleep(0.05)
            
            # Simular trabajo
            print(f"Trabajador {id}: Trabajando con recursos {recursos_necesarios}...")
            time.sleep(0.2)
            
        finally:
            # Liberar en orden inverso
            for idx in reversed(adquiridos):
                locks[idx].release()
                print(f"Trabajador {id}: Liberó {recursos[idx]}")
    
    # Crear trabajadores con necesidades de recursos que se solapan
    trabajadores = [
        (1, ["A", "B", "C"]),
        (2, ["C", "D", "E"]), 
        (3, ["A", "E", "B"]),
        (4, ["B", "D", "C"])
    ]
    
    threads = []
    for id, recursos_necesarios in trabajadores:
        t = threading.Thread(target=trabajador, args=(id, recursos_necesarios))
        threads.append(t)
        t.start()
        time.sleep(0.1)  # Espaciar el inicio
    
    # Verificar deadlock después de un tiempo
    time.sleep(3)
    vivos = sum(1 for t in threads if t.is_alive())
    print(f"\nTrabajadores aún vivos: {vivos}/{len(threads)}")
    
    if vivos > 0:
        print("¡INTERBLOQUEO DETECTADO! Algunos trabajadores están bloqueados.")

# =============================================================================
# 3. INTERBLOQUEO CON RECURSOS ANIDADOS (RLock)
# =============================================================================

def ejemplo_deadlock_recursivo():
    print("\n=== 3. INTERBLOQUEO CON LLAMADAS RECURSIVAS ===")
    
    lock = RLock()  # Lock recursivo
    
    def funcion_a():
        print("Función A: Adquiriendo lock...")
        with lock:
            print("Función A: Lock adquirido, llamando a función B...")
            funcion_b()
        print("Función A: Lock liberado")
    
    def funcion_b():
        print("Función B: Intentando adquirir lock...")
        with lock:  # ¡ESPERANDO! El lock ya está adquirido por A
            print("Función B: Lock adquirido")
            print("Función B: Haciendo trabajo...")
        print("Función B: Lock liberado")
    
    # Este ejemplo muestra que incluso con RLock puede haber deadlock
    # si no se maneja adecuadamente la recursividad
    print("Ejecutando función A (causará interbloqueo)...")
    
    t = threading.Thread(target=funcion_a)
    t.start()
    
    time.sleep(1)
    print(f"Hilo vivo después de 1 segundo: {t.is_alive()}")
    
    if t.is_alive():
        print("¡INTERBLOQUEO RECURSIVO DETECTADO!")

# =============================================================================
# 4. DETECCIÓN Y PREVENCIÓN DE INTERBLOQUEOS
# =============================================================================

def ejemplo_prevencion_deadlock():
    print("\n=== 4. PREVENCIÓN DE INTERBLOQUEOS ===")
    
    lock_a = Lock()
    lock_b = Lock()
    
    # Context manager para adquirir múltiples locks de forma segura
    @contextmanager
    def adquirir_locks(*locks):
        # Ordenar locks por su dirección de memoria para evitar deadlock
        locks_ordenados = sorted(locks, key=id)
        
        try:
            for lock in locks_ordenados:
                lock.acquire()
            yield
        finally:
            # Liberar en orden inverso
            for lock in reversed(locks_ordenados):
                lock.release()
    
    def hilo_seguro(id):
        print(f"Hilo {id}: Intentando adquirir locks de forma segura...")
        
        with adquirir_locks(lock_a, lock_b):
            print(f"Hilo {id}: Locks adquiridos exitosamente")
            print(f"Hilo {id}: Trabajando con los recursos...")
            time.sleep(0.2)
        
        print(f"Hilo {id}: Locks liberados")
    
    # Crear hilos que usan la prevención
    threads = []
    for i in range(3):
        t = threading.Thread(target=hilo_seguro, args=(i+1,))
        threads.append(t)
        t.start()
        time.sleep(0.1)
    
    # Esperar a que todos terminen
    for t in threads:
        t.join()
    
    print("Todos los hilos completaron sin interbloqueos!")

# =============================================================================
# 5. HERRAMIENTA DE DETECCIÓN DE INTERBLOQUEOS
# =============================================================================

def detector_interbloqueos():
    print("\n=== 5. DETECTOR DE INTERBLOQUEOS ===")
    
    class Recurso:
        def __init__(self, nombre):
            self.nombre = nombre
            self.lock = Lock()
            self.propietario = None
        
        def adquirir(self, propietario):
            print(f"{propietario}: Esperando {self.nombre}...")
            self.lock.acquire()
            self.propietario = propietario
            print(f"{propietario}: Adquirió {self.nombre}")
        
        def liberar(self):
            if self.propietario:
                print(f"{self.propietario}: Liberando {self.nombre}")
                self.propietario = None
            self.lock.release()
    
    recurso_x = Recurso("X")
    recurso_y = Recurso("Y")
    
    # Grafo de espera para detección de deadlocks
    grafo_espera = {}
    
    def actualizar_grafo(hilo, recurso_esperando, recurso_poseido=None):
        if hilo not in grafo_espera:
            grafo_espera[hilo] = {"espera": None, "posee": []}
        
        if recurso_esperando:
            grafo_espera[hilo]["espera"] = recurso_esperando
        
        if recurso_poseido:
            grafo_espera[hilo]["posee"].append(recurso_poseido)
    
    def verificar_deadlock():
        print("Verificando posible deadlock...")
        # Simulación simple de detección
        for hilo, info in grafo_espera.items():
            if info["espera"] and info["posee"]:
                recurso_esperado = info["espera"]
                print(f"{hilo} espera {recurso_esperado} mientras posee {info['posee']}")
    
    def hilo_con_deteccion(id, recurso1, recurso2):
        nombre_hilo = f"Hilo-{id}"
        print(f"{nombre_hilo}: Iniciando")
        
        try:
            # Adquirir primer recurso
            recurso1.adquirir(nombre_hilo)
            actualizar_grafo(nombre_hilo, None, recurso1.nombre)
            time.sleep(0.1)
            
            # Intentar adquirir segundo recurso
            actualizar_grafo(nombre_hilo, recurso2.nombre)
            recurso2.adquirir(nombre_hilo)
            actualizar_grafo(nombre_hilo, None, recurso2.nombre)
            
            # Trabajar
            print(f"{nombre_hilo}: Trabajando con ambos recursos...")
            time.sleep(0.2)
            
        finally:
            # Liberar recursos
            recurso2.liberar()
            recurso1.liberar()
            print(f"{nombre_hilo}: Completado")
    
    # Crear hilos que causarán deadlock
    t1 = threading.Thread(target=hilo_con_deteccion, args=(1, recurso_x, recurso_y))
    t2 = threading.Thread(target=hilo_con_deteccion, args=(2, recurso_y, recurso_x))
    
    t1.start()
    t2.start()
    
    # Verificar deadlock después de un tiempo
    time.sleep(1)
    verificar_deadlock()
    
    if t1.is_alive() and t2.is_alive():
        print("¡DEADLOCK CONFIRMADO POR EL DETECTOR!")

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("DEMOSTRACIÓN DE INTERBLOQUEOS EN ALGORITMOS PARALELOS")
    print("=" * 60)
    
    try:
        # Ejecutar ejemplos
        ejemplo_deadlock_clasico()
        time.sleep(2)
        
        ejemplo_deadlock_multiple()
        time.sleep(2)
        
        ejemplo_deadlock_recursivo()
        time.sleep(2)
        
        ejemplo_prevencion_deadlock()
        time.sleep(2)
        
        detector_interbloqueos()
        
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario")
    
    print("\n" + "=" * 60)
    print("CONCLUSIÓN:")
    print("- Los interbloqueos ocurren cuando hilos esperan recursos circularmente")
    print("- Se pueden prevenir mediante:")
    print("  * Ordenamiento consistente de adquisición de locks")
    print("  * Timeouts en las adquisiciones")
    print("  * Diseño cuidadoso de la estrategia de locking")
    print("  * Uso de detectores de deadlock")