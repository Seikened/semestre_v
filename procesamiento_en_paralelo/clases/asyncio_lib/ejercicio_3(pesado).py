import asyncio
import random
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor

tabla_uso = {
    'A': {'tiempo': 0.0, 'conteo': 0},
    'B': {'tiempo': 0.0, 'conteo': 0},
    'C': {'tiempo': 0.0, 'conteo': 0},
}

PROC_POOL = None  # se inicializa en main()

def estadisticas(obj, tiempo):
    obj['tiempo'] += tiempo
    obj['conteo'] += 1

# --- Trabajo CPU-BOUND simulado (sustituye por tu cómputo real) ---
def tarea_pesada_cpu(intensidad: int) -> int:
    """Carga CPU: suma de primos hasta un límite que depende de 'intensidad'."""
    limite = 200_000 * intensidad
    acc = 0
    for n in range(2, limite):
        is_prime = True
        r = int(n ** 0.5)
        for d in range(2, r + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            acc += n
    return acc

async def tarea_a(end_time):
    loop = asyncio.get_running_loop()

    # En vez de "esperar = randint(...)" para dormir,
    # usa "intensidad" para tu carga CPU.
    intensidad = random.randint(1, 5)
    print(f'Tarea [A] 🟢 Inicio, CPU intensidad {intensidad}')

    # Mide tiempo real de la sección pesada
    t0 = perf_counter()
    # Ejecuta trabajo pesado en procesos (paralelismo real)
    _ = await loop.run_in_executor(PROC_POOL, tarea_pesada_cpu, intensidad)
    dt = perf_counter() - t0

    # Registra tiempo consumido por A tras COMPLETAR el cómputo
    estadisticas(tabla_uso['A'], dt)

    if (loop.time() + 1.0) < end_time:
        asyncio.create_task(tarea_b(end_time))
    else:
        print('Tarea [A] finalizando loop')

async def tarea_b(end_time):
    loop = asyncio.get_running_loop()
    esperar = random.randint(1, 5)
    print(f'Tarea [B] 🔵 Inicio, esperando {esperar} segundos')
    # registra cuando realmente termina B
    await asyncio.sleep(esperar)
    estadisticas(tabla_uso['B'], float(esperar))
    if (loop.time() + 1.0) < end_time:
        asyncio.create_task(tarea_c(end_time))
    else:
        print('Tarea [B] finalizando loop')

async def tarea_c(end_time):
    loop = asyncio.get_running_loop()
    esperar = random.randint(1, 5)
    print(f'Tarea [C] 🟣 Inicio, esperando {esperar} segundos')
    await asyncio.sleep(esperar)
    estadisticas(tabla_uso['C'], float(esperar))
    if (loop.time() + 1.0) < end_time:
        asyncio.create_task(tarea_a(end_time))
    else:
        print('Tarea [C] finalizando loop')

async def main():
    random.seed()
    loop = asyncio.get_running_loop()
    duracion = 60
    end_time = loop.time() + duracion

    # Crea un ProcessPool con pocos workers (ajusta según tu máquina)
    global PROC_POOL
    PROC_POOL = ProcessPoolExecutor(max_workers=4)

    inicio = perf_counter()
    K = 10
    for _ in range(K):
        asyncio.create_task(tarea_a(end_time))

    await asyncio.sleep(duracion + 1.0)

    # Cierra ordenado el pool de procesos
    PROC_POOL.shutdown(cancel_futures=True)

    # Reporte final
    for clave, valor in tabla_uso.items():
        print(f"Tarea {clave}: Tiempo total = {valor['tiempo']:.3f} s, Conteo = {valor['conteo']}")
    a = sum(tabla_uso[k]['tiempo'] for k in tabla_uso)
    print(f'Tiempo total acumulado de todas las tareas: {a:.3f} s')
    fin = perf_counter()
    print(f'Tiempo real transcurrido: {fin - inicio:.3f} s')

if __name__ == '__main__':
    asyncio.run(main())