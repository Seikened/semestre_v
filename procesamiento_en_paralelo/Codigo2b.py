import numpy as np
import threading
import time
from queue import Queue


def multiplicacion_matrices_secuencial(A, B):
    """
    Multiplicación de matrices de forma secuencial (tradicional)

    Args:
        A: Matriz m x n
        B: Matriz n x p

    Returns:
        C: Matriz m x p resultante de A * B
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            "Las dimensiones de las matrices no son compatibles para multiplicación"
        )

    m, n = A.shape
    n, p = B.shape

    # Inicializar matriz resultado con ceros
    C = np.zeros((m, p))

    # Multiplicación secuencial
    for i in range(m):
        for j in range(p):
            suma = 0
            for k in range(n):
                suma += A[i, k] * B[k, j]
            C[i, j] = suma

    return C


def crear_matriz_aleatoria(filas, columnas, min_val=0, max_val=10):
    """Crea una matriz con valores aleatorios"""
    return np.random.randint(min_val, max_val, size=(filas, columnas))


def prueba_secuencial():
    """Prueba de la implementación secuencial"""
    print("🧪 PRUEBA SECUENCIAL")
    print("=" * 50)

    # Crear matrices de prueba
    tamaño = 1000  # Reducido para pruebas más rápidas
    A = crear_matriz_aleatoria(tamaño, tamaño)
    B = crear_matriz_aleatoria(tamaño, tamaño)

    print(f"📊 Matriz A: {A.shape}")
    print(f"📊 Matriz B: {B.shape}")

    # Ejecutar multiplicación secuencial
    print("\n⏳ Ejecutando multiplicación secuencial...")
    inicio = time.time()

    C_secuencial = multiplicacion_matrices_secuencial(A, B)

    tiempo_secuencial = time.time() - inicio
    print(f"✅ Tiempo secuencial: {tiempo_secuencial:.4f} segundos")

    # Verificar con numpy (opcional)
    print("🔍 Verificando resultado con numpy...")
    C_numpy = np.dot(A, B)

    # Comprobar que los resultados son iguales
    son_iguales = np.allclose(C_secuencial, C_numpy)
    print(f"📋 Resultados iguales: {son_iguales}")

    if not son_iguales:
        print("⚠️  ¡Advertencia: Los resultados no coinciden!")

    return tiempo_secuencial, C_secuencial


class MultiplicacionThread(threading.Thread):
    """Hilo para calcular una porción de la multiplicación de matrices"""

    def __init__(self, thread_id, A, B, C, start_row, end_row, lock, result_queue):
        """
        Inicializa el hilo de multiplicación

        Args:
            thread_id: Identificador del hilo
            A, B: Matrices a multiplicar
            C: Matriz resultado (compartida)
            start_row: Fila inicial a procesar
            end_row: Fila final a procesar
            lock: Lock para acceso seguro
            result_queue: Cola para notificar finalización
        """
        super().__init__()
        self.thread_id = thread_id
        self.A = A
        self.B = B
        self.C = C
        self.start_row = start_row
        self.end_row = end_row
        self.lock = lock
        self.result_queue = result_queue

    def run(self):
        """Ejecuta la multiplicación para el rango de filas asignado"""
        try:
            m, n = self.A.shape
            n, p = self.B.shape

            # Calcular las filas asignadas a este hilo
            for i in range(self.start_row, self.end_row):
                for j in range(p):
                    suma = 0
                    for k in range(n):
                        suma += self.A[i, k] * self.B[k, j]

                    # Guardar resultado de manera segura
                    with self.lock:
                        self.C[i, j] = suma

            # Notificar que terminó
            self.result_queue.put((self.thread_id, self.start_row, self.end_row))

        except Exception as e:
            print(f"❌ Error en hilo {self.thread_id}: {e}")
            self.result_queue.put((self.thread_id, None, None))


def multiplicacion_matrices_threading(A, B, num_threads=4):
    """
    Multiplicación de matrices usando threading

    Args:
        A: Matriz m x n
        B: Matriz n x p
        num_threads: Número de hilos a usar

    Returns:
        C: Matriz m x p resultante de A * B
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            "Las dimensiones de las matrices no son compatibles para multiplicación"
        )

    m, n = A.shape
    n, p = B.shape

    # Inicializar matriz resultado
    C = np.zeros((m, p))

    # Crear lock para acceso seguro a la matriz resultado
    lock = threading.Lock()

    # Cola para recibir resultados de los hilos
    result_queue = Queue()

    # Calcular rango de filas para cada hilo
    filas_por_thread = m // num_threads
    threads = []

    print(f"🧵 Creando {num_threads} hilos...")
    print(f"📊 Filas por hilo: ~{filas_por_thread}")

    # Crear y iniciar hilos
    for i in range(num_threads):
        start_row = i * filas_por_thread
        end_row = (i + 1) * filas_por_thread

        # El último hilo toma las filas restantes
        if i == num_threads - 1:
            end_row = m

        thread = MultiplicacionThread(
            thread_id=i,
            A=A,
            B=B,
            C=C,
            start_row=start_row,
            end_row=end_row,
            lock=lock,
            result_queue=result_queue,
        )
        threads.append(thread)
        thread.start()
        print(f"   Hilo {i}: filas {start_row}-{end_row - 1}")

    # Esperar a que todos los hilos terminen
    print("\n⏳ Esperando que los hilos terminen...")
    threads_completados = 0

    while threads_completados < num_threads:
        try:
            thread_id, start, end = result_queue.get(timeout=1.0)
            if start is not None:
                print(f"✅ Hilo {thread_id} completó filas {start}-{end - 1}")
            threads_completados += 1
        except:
            # Timeout, verificar si algún hilo murió
            pass

    # Hacer join de todos los hilos
    for thread in threads:
        thread.join()

    return C


def prueba_threading():
    """Prueba de la implementación con threading"""
    print("\n" + "🧪 PRUEBA CON THREADING")
    print("=" * 50)

    # Crear matrices de prueba (más pequeñas para demo)
    tamaño = 1000
    A = crear_matriz_aleatoria(tamaño, tamaño)
    B = crear_matriz_aleatoria(tamaño, tamaño)

    print(f"📊 Matriz A: {A.shape}")
    print(f"📊 Matriz B: {B.shape}")

    # Probar con diferente número de hilos
    for num_threads in [16]:
        print(f"\n🔧 Probando con {num_threads} hilos...")

        inicio = time.time()
        C_threading = multiplicacion_matrices_threading(A, B, num_threads)
        tiempo_threading = time.time() - inicio

        print(f"✅ Tiempo con {num_threads} hilos: {tiempo_threading:.4f} segundos")

        # Verificar resultado
        C_numpy = np.dot(A, B)
        son_iguales = np.allclose(C_threading, C_numpy)
        print(f"📋 Resultados correctos: {son_iguales}")

        if not son_iguales:
            print("⚠️  ¡Advertencia: Los resultados no coinciden!")


def comparar_rendimiento():
    """Compara el rendimiento de ambas implementaciones"""
    print("\n" + "📊 COMPARATIVA DE RENDIMIENTO")
    print("=" * 50)

    # Tamaños de matriz para probar
    tamanos = [100, 200, 300]
    resultados = []

    for tamaño in tamanos:
        print(f"\n🔬 Probando con matrices {tamaño}x{tamaño}...")

        A = crear_matriz_aleatoria(tamaño, tamaño)
        B = crear_matriz_aleatoria(tamaño, tamaño)

        # Secuencial
        inicio = time.time()
        C_sec = multiplicacion_matrices_secuencial(A, B)
        tiempo_sec = time.time() - inicio

        # Threading (4 hilos)
        inicio = time.time()
        C_thr = multiplicacion_matrices_threading(A, B, 4)
        tiempo_thr = time.time() - inicio

        # Verificar
        correcto = np.allclose(C_sec, C_thr)

        resultados.append(
            {
                "tamaño": tamaño,
                "secuencial": tiempo_sec,
                "threading": tiempo_thr,
                "speedup": tiempo_sec / tiempo_thr if tiempo_thr > 0 else 0,
                "correcto": correcto,
            }
        )

        print(f"   Secuencial: {tiempo_sec:.4f}s")
        print(f"   Threading:  {tiempo_thr:.4f}s")
        print(f"   Speedup:    {resultados[-1]['speedup']:.2f}x")
        print(f"   Correcto:   {correcto}")

    # Mostrar resumen
    print("\n" + "📈 RESUMEN DE RESULTADOS")
    print("=" * 50)
    for res in resultados:
        print(f"Matriz {res['tamaño']}x{res['tamaño']}:")
        print(f"  Secuencial: {res['secuencial']:.4f}s")
        print(f"  Threading:  {res['threading']:.4f}s")
        print(f"  Speedup:    {res['speedup']:.2f}x")
        print(f"  Correcto:   {res['correcto']}")
        print()


def explicacion_conceptos():
    """Explica los conceptos de threading usados"""
    print("\n" + "📚 CONCEPTOS DE THREADING APLICADOS")
    print("=" * 50)

    conceptos = [
        "🧵 DIVISIÓN DEL TRABAJO:",
        "   • Cada hilo procesa un grupo de filas de la matriz",
        "   • El trabajo se divide equitativamente entre hilos",
        "",
        "🔒 SINCRONIZACIÓN:",
        "   • Lock para acceso seguro a la matriz resultado compartida",
        "   • Previene condiciones de carrera al escribir resultados",
        "",
        "📨 COMUNICACIÓN:",
        "   • Queue para que los hilos notifiquen finalización",
        "   • El hilo principal monitorea el progreso",
        "",
        "⚡ PARALELISMO:",
        "   • Múltiples hilos ejecutándose simultáneamente",
        "   • Ideal para operaciones intensivas de CPU",
        "",
        "⚠️ LIMITACIONES:",
        "   • GIL (Global Interpreter Lock) en Python",
        "   • El threading en Python es mejor para I/O-bound que CPU-bound",
        "   • Para CPU-bound intensivo, considerar multiprocessing",
    ]

    for concepto in conceptos:
        print(concepto)


if __name__ == "__main__":
    # Ejecutar pruebas
    print("🎯 MULTIPLICACIÓN DE MATRICES: SECUENCIAL vs THREADING")
    print("=" * 60)

    # Prueba secuencial
    # prueba_secuencial()

    # Prueba threading
    prueba_threading()

    # Comparativa de rendimiento
    # comparar_rendimiento()

    # Explicación de conceptos
    #input("\nPresiona Enter para ver la explicación de conceptos...")
    explicacion_conceptos()

    print("\n" + "🎉 PRUEBAS COMPLETADAS!")
