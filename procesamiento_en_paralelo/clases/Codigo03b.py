import multiprocessing as mp
import time
import random
from multiprocessing import Lock, RLock, Manager
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(message)s')

# =============================================================================
# 1. INTERBLOQUEO CLÁSICO CON MULTIPROCESSING (CORREGIDO)
# =============================================================================

def ejemplo_deadlock_clasico():
    """Deadlock clásico con dos procesos y dos locks - CORREGIDO"""
    print("=== 1. INTERBLOQUEO CLÁSICO CON MULTIPROCESSING ===")
    
    # Crear locks usando Manager para compartirlos entre procesos
    manager = Manager()
    lock_a = manager.Lock()
    lock_b = manager.Lock()
    
    def proceso_1(lock_a, lock_b):
        pid = os.getpid()
        logging.info(f"Proceso 1 (PID: {pid}): Adquiriendo Lock A...")
        lock_a.acquire()
        logging.info(f"Proceso 1: Lock A adquirido")
        time.sleep(0.5)  # Simular trabajo
        
        logging.info(f"Proceso 1: Intentando adquirir Lock B...")
        lock_b.acquire()  # ¡ESPERANDO!
        logging.info(f"Proceso 1: Lock B adquirido")
        
        # Trabajo con ambos recursos
        logging.info(f"Proceso 1: Trabajando con ambos locks...")
        time.sleep(0.3)
        
        lock_b.release()
        lock_a.release()
        logging.info(f"Proceso 1: Locks liberados")
    
    def proceso_2(lock_a, lock_b):
        pid = os.getpid()
        logging.info(f"Proceso 2 (PID: {pid}): Adquiriendo Lock B...")
        lock_b.acquire()
        logging.info(f"Proceso 2: Lock B adquirido")
        time.sleep(0.5)  # Simular trabajo
        
        logging.info(f"Proceso 2: Intentando adquirir Lock A...")
        lock_a.acquire()  # ¡ESPERANDO!
        logging.info(f"Proceso 2: Lock A adquirido")
        
        # Trabajo con ambos recursos
        logging.info(f"Proceso 2: Trabajando con ambos locks...")
        time.sleep(0.3)
        
        lock_a.release()
        lock_b.release()
        logging.info(f"Proceso 2: Locks liberados")
    
    try:
        # Crear procesos pasando los locks como argumentos
        p1 = mp.Process(target=proceso_1, args=(lock_a, lock_b), name="Proceso-1")
        p2 = mp.Process(target=proceso_2, args=(lock_a, lock_b), name="Proceso-2")
        
        p1.start()
        p2.start()
        
        # Monitorear el estado
        for i in range(3):
            time.sleep(1)
            print(f"\nEstado después de {i+1} segundos:")
            print(f"Proceso 1 vivo: {p1.is_alive()}, Proceso 2 vivo: {p2.is_alive()}")
            
            if p1.is_alive() and p2.is_alive():
                print("¡INTERBLOQUEO DETECTADO! Ambos procesos bloqueados")
        
    finally:
        # Asegurar que los procesos terminen
        if p1.is_alive():
            p1.terminate()
        if p2.is_alive():
            p2.terminate()
        
        p1.join(timeout=1)
        p2.join(timeout=1)
        
        # Cerrar el manager explícitamente después de que los procesos terminen
        manager.shutdown()

# =============================================================================
# 2. INTERBLOQUEO CON MÚLTIPLES PROCESOS Y RECURSOS (CORREGIDO)
# =============================================================================

def ejemplo_deadlock_multiple():
    """Deadlock con múltiples procesos compitiendo por recursos - CORREGIDO"""
    print("\n=== 2. INTERBLOQUEO CON MÚLTIPLES PROCESOS ===")
    
    manager = Manager()
    
    try:
        # Crear múltiples locks compartidos
        num_recursos = 4
        locks = [manager.Lock() for _ in range(num_recursos)]
        nombres_recursos = ["Recurso-A", "Recurso-B", "Recurso-C", "Recurso-D"]
        
        def trabajador(id, recursos_necesarios, locks, nombres_recursos):
            pid = os.getpid()
            logging.info(f"Trabajador {id} (PID: {pid}): Necesita {recursos_necesarios}")
            
            adquiridos = []
            try:
                # Adquirir recursos en orden aleatorio (puede causar deadlock)
                for recurso in recursos_necesarios:
                    idx = nombres_recursos.index(recurso)
                    logging.info(f"Trabajador {id}: Esperando {recurso}...")
                    locks[idx].acquire()
                    adquiridos.append(idx)
                    logging.info(f"Trabajador {id}: Adquirió {recurso}")
                    time.sleep(0.2)
                
                # Trabajar con los recursos
                logging.info(f"Trabajador {id}: Procesando con {recursos_necesarios}...")
                time.sleep(random.uniform(0.3, 0.7))
                
            finally:
                # Liberar recursos
                for idx in reversed(adquiridos):
                    try:
                        locks[idx].release()
                        logging.info(f"Trabajador {id}: Liberó {nombres_recursos[idx]}")
                    except:
                        logging.warning(f"Trabajador {id}: Error liberando {nombres_recursos[idx]}")
        
        # Configurar trabajadores con necesidades que se solapan
        configuraciones = [
            (1, ["Recurso-A", "Recurso-B"]),
            (2, ["Recurso-B", "Recurso-C"]),
            (3, ["Recurso-C", "Recurso-D"]),
            (4, ["Recurso-D", "Recurso-A"]),  # Este crea el ciclo de deadlock
        ]
        
        procesos = []
        for id, recursos in configuraciones:
            p = mp.Process(target=trabajador, 
                          args=(id, recursos, locks, nombres_recursos),
                          name=f"Worker-{id}")
            procesos.append(p)
            p.start()
            time.sleep(0.1)  # Espaciar inicio
        
        # Monitorear
        time.sleep(2)
        vivos = [p for p in procesos if p.is_alive()]
        print(f"\nProcesos bloqueados: {len(vivos)}/{len(procesos)}")
        
        if vivos:
            print("¡INTERBLOQUEO MULTIPLE DETECTADO!")
            for p in vivos:
                print(f"  - {p.name} aún bloqueado")
        
    finally:
        # Limpiar
        for p in procesos:
            if p.is_alive():
                p.terminate()
            p.join(timeout=1)
        
        manager.shutdown()

# =============================================================================
# 3. INTERBLOQUEO CON COLA DE MENSAJES (CORREGIDO)
# =============================================================================

def ejemplo_deadlock_colas():
    """Deadlock con procesos esperando mensajes entre sí - CORREGIDO"""
    print("\n=== 3. INTERBLOQUEO CON COMUNICACIÓN ENTRE PROCESOS ===")
    
    manager = Manager()
    
    try:
        cola = manager.Queue()
        lock_compartido = manager.Lock()
        
        def proceso_productor(cola, lock, id_proceso):
            """Proceso que produce datos pero necesita confirmación"""
            pid = os.getpid()
            logging.info(f"Productor {id_proceso} (PID: {pid}): Iniciando")
            
            with lock:
                logging.info(f"Productor {id_proceso}: Enviando mensaje...")
                cola.put(f"Mensaje de {id_proceso}")
                logging.info(f"Productor {id_proceso}: Esperando confirmación...")
            
            # Esperar confirmación (que nunca llegará)
            try:
                respuesta = cola.get(timeout=3)
                logging.info(f"Productor {id_proceso}: Recibió {respuesta}")
            except:
                logging.warning(f"Productor {id_proceso}: Timeout esperando respuesta")
        
        def proceso_consumidor(cola, lock, id_proceso):
            """Proceso que consume datos pero también quiere enviar"""
            pid = os.getpid()
            logging.info(f"Consumidor {id_proceso} (PID: {pid}): Iniciando")
            
            # Primero intentar enviar algo
            with lock:
                logging.info(f"Consumidor {id_proceso}: Intentando enviar solicitud...")
                cola.put(f"Solicitud de {id_proceso}")
            
            # Luego intentar recibir
            try:
                mensaje = cola.get(timeout=3)
                logging.info(f"Consumidor {id_proceso}: Procesando {mensaje}")
            except:
                logging.warning(f"Consumidor {id_proceso}: Timeout esperando mensaje")
        
        # Crear procesos que se esperan mutuamente
        productor = mp.Process(target=proceso_productor, 
                              args=(cola, lock_compartido, 1),
                              name="Productor")
        consumidor = mp.Process(target=proceso_consumidor, 
                               args=(cola, lock_compartido, 2),
                               name="Consumidor")
        
        productor.start()
        consumidor.start()
        
        time.sleep(2)
        
        if productor.is_alive() and consumidor.is_alive():
            print("¡INTERBLOQUEO EN COMUNICACIÓN DETECTADO!")
            print("Ambos procesos esperando mensajes del otro")
        
    finally:
        # Asegurar terminación
        if 'productor' in locals() and productor.is_alive():
            productor.terminate()
        if 'consumidor' in locals() and consumidor.is_alive():
            consumidor.terminate()
        
        if 'productor' in locals():
            productor.join(timeout=1)
        if 'consumidor' in locals():
            consumidor.join(timeout=1)
        
        manager.shutdown()

# =============================================================================
# 4. INTERBLOQUEO CON RECURSOS COMPARTIDOS Y CONDICIONES (CORREGIDO)
# =============================================================================

def ejemplo_deadlock_condiciones():
    """Deadlock con condiciones y wait/notify - CORREGIDO"""
    print("\n=== 4. INTERBLOQUEO CON VARIABLES DE CONDICIÓN ===")
    
    manager = Manager()
    
    try:
        cond = manager.Condition(manager.Lock())
        recurso_disponible = manager.Value('b', False)  # Recurso no disponible inicialmente
        
        def proceso_condicion(cond, recurso, id_proceso, necesita_recurso):
            """Proceso que espera por una condición"""
            pid = os.getpid()
            logging.info(f"Proceso {id_proceso} (PID: {pid}): Iniciando")
            
            with cond:
                if necesita_recurso and not recurso.value:
                    logging.info(f"Proceso {id_proceso}: Esperando recurso...")
                    cond.wait(timeout=2)  # Esperar notificación
                    
                    if not recurso.value:
                        logging.warning(f"Proceso {id_proceso}: Timeout en espera")
                        return
                
                # Trabajar con recurso
                logging.info(f"Proceso {id_proceso}: Usando recurso")
                time.sleep(0.5)
                
                if not necesita_recurso:
                    recurso.value = True
                    logging.info(f"Proceso {id_proceso}: Liberando recurso")
                    cond.notify_all()
        
        # Proceso que necesita el recurso
        p1 = mp.Process(target=proceso_condicion, 
                       args=(cond, recurso_disponible, 1, True),
                       name="Consumidor")
        
        # Proceso que debería liberar el recurso pero también espera
        p2 = mp.Process(target=proceso_condicion,
                       args=(cond, recurso_disponible, 2, False),
                       name="Productor")
        
        p1.start()
        time.sleep(0.1)
        p2.start()
        
        time.sleep(3)
        
        if p1.is_alive():
            print("¡INTERBLOQUEO CON CONDICIONES DETECTADO!")
            print("Proceso consumidor esperando indefinidamente")
        
    finally:
        # Limpiar
        if 'p1' in locals() and p1.is_alive():
            p1.terminate()
        if 'p2' in locals() and p2.is_alive():
            p2.terminate()
        
        if 'p1' in locals():
            p1.join(timeout=1)
        if 'p2' in locals():
            p2.join(timeout=1)
        
        manager.shutdown()

# =============================================================================
# 5. PREVENCIÓN DE INTERBLOQUEOS (CORREGIDO)
# =============================================================================

def ejemplo_prevencion_deadlock():
    """Técnicas para prevenir deadlocks en multiprocessing - CORREGIDO"""
    print("\n=== 5. PREVENCIÓN DE INTERBLOQUEOS ===")
    
    manager = Manager()
    
    try:
        # Crear locks con nombres
        locks = {
            'A': manager.Lock(),
            'B': manager.Lock(),
            'C': manager.Lock()
        }
        
        def trabajador_seguro(id, locks, orden_locks, timeout=2):
            """Worker con prevención de deadlock usando timeouts"""
            pid = os.getpid()
            logging.info(f"Trabajador Seguro {id} (PID: {pid}): Iniciando")
            
            adquiridos = []
            try:
                # Adquirir locks en orden consistente
                for lock_name in orden_locks:
                    lock = locks[lock_name]
                    logging.info(f"Trabajador {id}: Intentando {lock_name} con timeout...")
                    
                    if lock.acquire(timeout=timeout):
                        adquiridos.append(lock_name)
                        logging.info(f"Trabajador {id}: Adquirió {lock_name}")
                        time.sleep(0.1)
                    else:
                        logging.warning(f"Trabajador {id}: Timeout en {lock_name}")
                        # Liberar todos los locks adquiridos
                        for adquirido in reversed(adquiridos):
                            locks[adquirido].release()
                        return False
                
                # Trabajar
                logging.info(f"Trabajador {id}: Procesando con {adquiridos}...")
                time.sleep(0.3)
                return True
                
            finally:
                # Liberar en orden inverso
                for lock_name in reversed(adquiridos):
                    try:
                        locks[lock_name].release()
                        logging.info(f"Trabajador {id}: Liberó {lock_name}")
                    except:
                        pass  # Ignorar errores al liberar
        
        # Procesos con órdenes consistentes de adquisición
        procesos = []
        configuraciones = [
            (1, ['A', 'B', 'C']),
            (2, ['A', 'B', 'C']),  # Mismo orden = sin deadlock
            (3, ['A', 'C']),
        ]
        
        for id, orden in configuraciones:
            p = mp.Process(target=trabajador_seguro, 
                          args=(id, locks, orden),
                          name=f"Seguro-{id}")
            procesos.append(p)
            p.start()
            time.sleep(0.1)
        
        # Esperar y verificar
        time.sleep(2)
        exitosos = sum(1 for p in procesos if p.exitcode == 0)
        print(f"\nProcesos exitosos: {exitosos}/{len(procesos)}")
        
    finally:
        # Limpiar
        for p in procesos:
            if p.is_alive():
                p.terminate()
            p.join(timeout=1)
        
        manager.shutdown()

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("DEMOSTRACIÓN DE INTERBLOQUEOS EN MULTIPROCESSING (CORREGIDO)")
    print("=" * 70)
    
    try:
        # Ejecutar ejemplos en orden
        ejemplo_deadlock_clasico()
        time.sleep(2)
        
        ejemplo_deadlock_multiple()
        time.sleep(2)
        
        ejemplo_deadlock_colas()
        time.sleep(2)
        
        ejemplo_deadlock_condiciones()
        time.sleep(2)
        
        ejemplo_prevencion_deadlock()
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario")
    
    print("\n" + "=" * 70)
    print("CORRECCIONES APLICADAS:")
    print("1. Manager creado y cerrado explícitamente en cada función")
    print("2. Los locks se pasan como argumentos a los procesos")
    print("3. Manejo adecuado de excepciones con bloques try-finally")
    print("4. Timeouts en las operaciones join()")
    print("5. Terminación controlada de procesos bloqueados")