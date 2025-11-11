import asyncio
import time
import random


tabla_uso = {
    'A': 0,
    'B': 0,
    'C': 0,
}


def tarea_a(end_time, loop):
    esperar = random.randint(1, 5)
    print(f'Tarea [A] 🟢 Inicio, esperando {esperar} segundos')
    tabla_uso['A'] += 1
    time.sleep(esperar)
    if (loop.time() + 1.0) < end_time:
        loop.call_later(1.0, tarea_b, end_time, loop)
    else:
        loop.stop()
        
def tarea_b(end_time, loop):
    esperar = random.randint(1, 5)
    print(f'Tarea [B] 🔵 Inicio, esperando {esperar} segundos')
    tabla_uso['B'] += 1
    time.sleep(esperar)
    if (loop.time() + 1.0) < end_time:
        loop.call_later(1.0, tarea_c, end_time, loop)
    else:
        loop.stop()
    
def tarea_c(end_time, loop):
    esperar = random.randint(1, 5)
    print(f'Tarea [C] 🟣 Inicio, esperando {esperar} segundos')
    tabla_uso['C'] += 1
    time.sleep(esperar)
    if (loop.time() + 1.0) < end_time:
        loop.call_later(1.0, tarea_a, end_time, loop)
    else:
        loop.stop()
        

if __name__ == '__main__':
    # Obtener el loop de eventos
    loop = asyncio.get_event_loop()
    
    # Definir el tiempo de finalización (60 segundos desde ahora)
    end_loop = loop.time() + 60.0
    
    # Establecer llamada a la tarea A
    loop.call_soon(tarea_a, end_loop, loop)
    
    # Iniciar el loop de eventos
    loop.run_forever()
    
    
    
    loop.close()
    # Mostrar tabla de uso
    for tarea, uso in tabla_uso.items():
        print(f'Tarea [{tarea}] fue ejecutada {uso} veces.')
        
    