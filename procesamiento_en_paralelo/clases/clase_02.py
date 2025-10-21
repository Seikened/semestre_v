from mpi4py import MPI
import numpy as np

def paramretros():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #print(f"Soy el proceso {rank} de un total de {size} procesos")
    return comm, rank, size


def topologias():
    
    # Definir constantes para las direcciones
    ARRIBA = 0
    ABAJO = 1
    IZQUIERDA = 2
    DERECHA = 3
    
    # Nombres de direcciones para mostrar en la salida
    NOMBRES_DIRECCIONES = ['ARRIBA', 'ABAJO', 'IZQUIERDA', 'DERECHA']
    
    
    # Inicializar MPI 
    comm, rank, total_procesos = paramretros()
    
    
    # Calcular dimensiones de la malla
    filas_malla = int(np.floor(np.sqrt(total_procesos)))
    columnas_malla = total_procesos // filas_malla
    
    
    # Ajustar dimesiones si no coinciden exactamente
    if filas_malla * columnas_malla > total_procesos:
        columnas_malla -= 1
    if filas_malla * columnas_malla > total_procesos:
        filas_malla -= 1
    
    
    v = 20
    if rank == 0:
        print(f"{"=" * v}")
        print(f"Topología de Malla 2D con {total_procesos} ")
        print(f"{"=" * v}")
        print(f"Creando malla de Filas: {filas_malla} Columnas: {columnas_malla}")
        print("Cada proceso tendra 4 vecinos (con condiciones periodicas)")
        print(f"{"=" * v}")

    # Crear la topología cartesiana
    comunicador_cartesiano = comm.Create_cart(
        dims=(filas_malla, columnas_malla),
        periods=(True, True),
        reorder=True
    )


    # Obtener las coordenadas del proceso actual
    fila_actual, columna_actual = comunicador_cartesiano.Get_coords(rank)

    # Inicializar lista para almacenar vecinos
    vecinos = [MPI.PROC_NULL] * 4
    
    # Encontrar vecinos en la dirección vertical (dimensión 0)
    vecinos[ARRIBA], vecinos[ABAJO] = comunicador_cartesiano.Shift(
        direction=0,
        disp=1
    )
    
    # Encontrar vecinos en la dirección horizontal (dimensión 1)
    vecinos[IZQUIERDA], vecinos[DERECHA] = comunicador_cartesiano.Shift(
        direction=1,
        disp=1
    )
    
    # Mostrar información de conectividad para cada proceso
    print(f"Proceso {rank:2d} -> Posición ({fila_actual}, {columna_actual}) tiene vecinos:")
    for direccion in range(4):
        nombre = NOMBRES_DIRECCIONES[direccion]
        vecino = vecinos[direccion]
        
        if vecino == MPI.PROC_NULL:
            estado = "Sin vecino"
        else:
            estado = f"Vecino en proceso {vecino}"

        print(f"    {nombre:>10} : {estado}")
    print("=" * v)
    
    
    # PRUEBA 1 | Comunicación básica DERECHA-IZQUIERDA
    if rank == 0:
        print("Prueba 1 | Comunicación básica DERECHA-IZQUIERDA")
        print("=" * v)
        
    comm.Barrier()
    # Usar send  para evitar deadlock
    if vecinos[DERECHA] != MPI.PROC_NULL and vecinos[IZQUIERDA] != MPI.PROC_NULL:
        mensaje_local = f"Mensaje desde proceso {rank}"
        mensaje_recibido = comunicador_cartesiano.sendrecv(
            sendobj = mensaje_local,
            dest = vecinos[DERECHA],
            sendtag = 100,
            source = vecinos[IZQUIERDA],
            recvtag = 100
        )

        print(f'Proceso {rank:2d} recibió: "{mensaje_recibido}" de proceso {vecinos[IZQUIERDA]} y envió a mensaje: "{mensaje_local}" a proceso {vecinos[DERECHA]}')
        
    comm.Barrier()

    # PRUEBA 2 | Comunicación básica ABAJO-ARRIBA
    if rank == 0:
        print("Prueba 2 | Comunicación básica ABAJO-ARRIBA")
        print("=" * v)
    comm.Barrier()
    
    if vecinos[ABAJO] != MPI.PROC_NULL and vecinos[ARRIBA] != MPI.PROC_NULL:
        mensaje_local = f"Mensaje desde proceso {rank}"
        mensaje_recibido = comunicador_cartesiano.sendrecv(
            sendobj = mensaje_local,
            dest = vecinos[ABAJO],
            sendtag = 200,
            source = vecinos[ARRIBA],
            recvtag = 200
        )

        print(f'Proceso {rank:2d} recibió: "{mensaje_recibido}" de proceso {vecinos[ARRIBA]} y envió a mensaje: "{mensaje_local}" a proceso {vecinos[ABAJO]}')

    comm.Barrier()
    


    # PRUEBA 3 | Intercambio de datos numéricos
    if rank == 0:
        print("Prueba 3 | Intercambio de datos numéricos")
        print("=" * v)
    comm.Barrier()
    
    # Intercambiar valores con los vecinos de la derecha

    if vecinos[DERECHA] != MPI.PROC_NULL and vecinos[IZQUIERDA] != MPI.PROC_NULL:
        mensaje_local = np.array([rank * 10 + fila_actual] , dtype=int)
        mensaje_recibido = np.array([-1], dtype=int)
        
        comunicador_cartesiano.Sendrecv(
            sendbuf = mensaje_local,
            dest = vecinos[DERECHA],
            sendtag = 300,
            recvbuf = mensaje_recibido,
            source = vecinos[IZQUIERDA],
            recvtag = 300
        )

        print(f'Proceso {rank:2d} recibió: "{mensaje_recibido}" de proceso {vecinos[IZQUIERDA]} y envió a mensaje: "{mensaje_local}" a proceso {vecinos[DERECHA]}')

    comm.Barrier()
    
    
    
    # PRUEBA 4 | Comunicación en las cuatro direcciones
    if rank == 0:
        print("Prueba 4 | Comunicación en las cuatro direcciones")
        print("=" * v)
    comm.Barrier()
    
    # Comunicación no bloqueante para evitar deadlock complejos
    requests = []
    
    # Enviar mensajes a todos los vecinos
    for direccion in range(4):
        if vecinos[direccion] != MPI.PROC_NULL:
            mensaje = f"Hola desde {rank} hacia {NOMBRES_DIRECCIONES[direccion]}"
            req = comunicador_cartesiano.isend(
                obj=mensaje,
                dest=vecinos[direccion],
                tag=400 + direccion
            )
            requests.append(req)

    # Esperar a que se completen todas las solicitudes
    mensajes_recibidos = []
    
    for direccion in range(4):
        if vecinos[direccion] != MPI.PROC_NULL:
            try:
                mensaje_recibido = comunicador_cartesiano.recv(
                    source=vecinos[direccion],
                    tag=400 + direccion
                )
                mensajes_recibidos.append(f" De {NOMBRES_DIRECCIONES[direccion]}: {mensaje_recibido}")
            except MPI.Exception as e:
                print(f"Error en la recepción de {NOMBRES_DIRECCIONES[direccion]}: {e}")


    # Esperar a que todos los envíos se completen
    MPI.Request.Waitall(requests)
    
    if mensajes_recibidos:
        print(f'Proceso {rank:2d} recibió mensajes:{",".join(mensajes_recibidos)}')
        print("=" * v)  
        
    
    comm.Barrier()
    
    
    
    
    if rank == 0:
        print("=" * v)  
        print("Ejemplo de uso:")
        print("Cada proceso puede comunicarse directamente con sus vecinos.")
        print("usando los ID's obtenidos, por ejemplo:")
        print("- comm.Send() y comm.Recv() con vecinos específicos.")
        print("Intercambiar datos en simulaciones")
        print("=" * v)  
    
    
    
    
    
if __name__ == "__main__":
    topologias()