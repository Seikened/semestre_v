import asyncio
import random

tabla_uso = {
    'A': {'tiempo': 0, 'conteo': 0},
    'B': {'tiempo': 0, 'conteo': 0},
    'C': {'tiempo': 0, 'conteo': 0},
}

def estadisticas(obj, tiempo):
    obj['tiempo'] += tiempo
    obj['conteo'] += 1

async def tarea_a(end_time):
    loop = asyncio.get_running_loop()
    esperar = random.randint(1, 5)
    print(f'Tarea [A] 🟢 Inicio, esperando {esperar} segundos')
    estadisticas(tabla_uso['A'], esperar)
    await asyncio.sleep(esperar)
    if (loop.time() + 1.0) < end_time:
        asyncio.create_task(tarea_b(end_time))
    else:
        print('Tarea [A] finalizando loop')

async def tarea_b(end_time):
    loop = asyncio.get_running_loop()
    esperar = random.randint(1, 5)
    print(f'Tarea [B] 🔵 Inicio, esperando {esperar} segundos')
    estadisticas(tabla_uso['B'], esperar)
    await asyncio.sleep(esperar)
    if (loop.time() + 1.0) < end_time:
        asyncio.create_task(tarea_c(end_time))
    else:
        print('Tarea [B] finalizando loop')

async def tarea_c(end_time):
    loop = asyncio.get_running_loop()
    esperar = random.randint(1, 5)
    print(f'Tarea [C] 🟣 Inicio, esperando {esperar} segundos')
    estadisticas(tabla_uso['C'], esperar)
    await asyncio.sleep(esperar)
    if (loop.time() + 1.0) < end_time:
        asyncio.create_task(tarea_a(end_time))
    else:
        print('Tarea [C] finalizando loop')

async def main():
    random.seed()  # opcional: fija una semilla si quieres reproducibilidad
    loop = asyncio.get_running_loop()
    duracion = 60
    end_time = loop.time() + duracion

    # **Cambio mínimo clave**: lanza varias cadenas A→B→C en paralelo.
    from time import perf_counter
    inicio = perf_counter()
    K = 10
    for _ in range(K):
        asyncio.create_task(tarea_a(end_time))

    # Mantén vivo el loop hasta pasar el end_time.
    await asyncio.sleep(duracion + 1.0)

    # Reporte final
    for clave, valor in tabla_uso.items():
        print(f"Tarea {clave}: Tiempo total = {valor['tiempo']} s, Conteo = {valor['conteo']}")
        
    # Tiempo total de ejecución 
    a = sum(tabla_uso[k]['tiempo'] for k in tabla_uso)
    print(f'Tiempo total de ejecución de todas las tareas: {a} segundos')
    fin = perf_counter()
    print(f'Tiempo real transcurrido: {fin - inicio} segundos')

if __name__ == '__main__':
    asyncio.run(main())