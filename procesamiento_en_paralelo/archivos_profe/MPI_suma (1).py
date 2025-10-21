from mpi4py import MPI
from os import environ
from typing import *

# Cree una variable de tipo para los tipos de objetos del comunicador MPI que podemos usar.
MPIComm = Union[MPI.Intracomm, MPI.Intercomm]

# Nuestra rutina principal. Comprueba el funcionamiento de MPI y 
# ejecuta el código correcto
def main():
    # type: () -> int
    """Executed when called via the CLI.
    Performs some sanity checks, and then calls the appropriate method.
    """

    # Obtener comunicador MPI, rango y el numero de procesos.
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # ¿Solo tenemos un proceso? Si es así, salgamos.
    if mpi_size == 1:
        print('You are running an MPI program with only one slot/task!')
        print('Are you using `mpiexec`?')
        print('If you are, then please use an `-n` of at least 2!')
        print('If you did all that, then your MPI setup may be bad.')
        return 1

    # ¿El tamaño de nuestro mundo es superior a 999? El resultado será un poco extraño.
    #  NOTA: Solo el rango cero tiene efecto, por lo que no obtenemos resultados duplicados.
    if mpi_size >= 1000 and mpi_rank == 0:
        print('WARNING:  Your world size {} is over 999!'.format(mpi_size))
        print("The output formatting will be a little weird, but that's it.")

    # ¡Comprobaciones de cordura completadas!

    # Llamar a la función apropiada, en función de nuestro rango
    if mpi_rank == 0:
        return mpi_root(mpi_comm)
    else:
        return mpi_nonroot(mpi_comm)

# Este programa tiene dos partes: El controlador y la parte del trabajador.
# El controlador es ejecutado por el rango 0; los trabajadores por todos los demás. 
# ALGUNAS TERMINOLOGÍAS: 
# Mundo MPI: Todos los procesos MPI, generados por `mpiexec`. 
# Tamaño MPI: El número de procesos MPI. 
# Rango MPI: Un número entero, en el rango [0, tamaño MPI).
# El rango MPI es único para cada trabajador. 
# Este programa sigue la convención de que el rango 0 es el "controlador", y todos
# los rangos distintos de cero son "trabajadores". Esto es importante cuando se usan cosas como 
# difusión o dispersión/recopilación. Pero si solo realizas envíos y recepciones simples (lo cual no hacemos),
# no necesitas ceñirte al paradigma controlador-trabajador. ¡Aun así, es una buena idea!

# Nuestro codigo para rank 0
def mpi_root(mpi_comm):
    # type: (MPIComm) -> int
    """Routine for the controller (MPI rank 0).

    Genera un número no negativo de 32 bits y lo difunde. 
    Luego, recopila las respuestas de todos los demás rangos MPI. 
    Cada rango MPI debe devolver una tupla de (cadena, entero). 
    La cadena es un "Identificador de CPU" MPI (normalmente un nombre de host). 
    El entero es el resultado de `random_number + MPI_rank`. 
    Una vez recopilados todos los resultados, genera cada uno (la matriz recopilada se ordena por rango MPI). 
    Verifica que cada entero devuelto sea correcto realizando el cálculo localmente
   (`returned int == random_number + MPI_rank`). 
   Finalmente, envía a cada trabajador (mediante un mensaje unicast) un `int` cero. 
   Luego, espera a que todos estén en el mismo punto del código (mediante un battier). 
   ¡Listo!
    """

    import random

    # Obtenga nuestro número aleatorio, transmítalo e imprímalo en la salida estándar. 
    # NOTA: Los métodos en minúsculas (como `bcast()`) toman el objeto Python y hacen 
    # la serialización por nosotros (¡genial!). 
    # `bast()` es bloqueante, en el sentido de que no retorna hasta 
    # que se hayan enviado los datos, pero _not_ está sincronizando. 
    # Tampoco hay garantía de _how_ se transmitieron los datos. 
    # NOTA: En Python 3.6+, deberíamos usar `secret` en lugar de `random`.

    random_number = random.randrange(2**32)
    mpi_comm.bcast(random_number)
    print('Controller @ MPI Rank   0:  Input {}'.format(random_number))

    # Recopila todas nuestras respuestas. `gather()` toma un parámetro, 
    # que para la raíz es `None`. 
    # Nuevamente, desde la perspectiva de MPI, esto es bloqueante y sincroniza,
    # en el sentido de que sabemos que todos los trabajadores han enviado algo.
    # Sin embargo, esto sigue sin ser una sincronización correcta. 
    # NOTA: La matriz devuelta coloca nuestro parámetro en la ranura 0. 
    # Por lo tanto, la longitud de la matriz devuelta debe coincidir con el tamaño del mundo MPI.
    GatherResponseType = List[Tuple[str, int]]
    response_array = mpi_comm.gather(None) # type: GatherResponseType

    # Comprobación de cordura: ¿Recibimos respuesta de todos?
    mpi_size = mpi_comm.Get_size()
    if len(response_array) != mpi_size:
        print('ERROR!  The MPI world has {} members, but we only gathered {}!'
            .format(mpi_size, len(response_array))
        )
        return 1

    # Output each worker's results.
    # NOTE: We skip entry zero, because rank 0 is us, and we gathered `None`.
    for i in range(1, mpi_size):
        # Sanity check: Did we get a tuple of (string, int)?
        if len(response_array[i]) != 2:
            print('WARNING!  MPI rank {} sent a mis-sized ({}) tuple!'
                .format(
                    i,
                    len(response_array[i])
                )
            )
            continue
        if type(response_array[i][0]) is not str:
            print('WARNING!  MPI rank {} sent a tuple with a {} instead of a str!'
                .format(
                    i,
                    str(type(response_array[i][0]))
                )
            )
            continue
        if type(response_array[i][1]) is not int:
            print('WARNING!  MPI rank {} sent a tuple with a {} instead of an int!'
                .format(
                    i,
                    str(type(response_array[i][1]))
                )
            )
            continue

        # Is the result OK?  Check if random_number + i == response
        if random_number + i == response_array[i][1]:
            result = 'OK'
        else:
            result = 'BAD'

        # Mostrar el mensaje. 
        # El primer campo `{: >3}` se traduce como... 
        # `: -> Toma el siguiente parámetro de `format()` # ` ` -> Usa espacios como carácter de relleno 
        # `>` -> Alineación a la derecha 
        # `3` -> El ancho normal es de tres caracteres
        print('   Worker at MPI Rank {: >3}: Output {} is {} (from {})'
            .format(
                i,
                response_array[i][1],
                result,
                response_array[i][0],
            )
        )

        # Envía al trabajador un mensaje MPI unidireccional que indica 
        # "¡Puedes salir ahora!". 
        # En realidad, esto solo muestra la mensajería MPI unidireccional. 
        # NOTA: ¡Esto será lento! Ya que tenemos que:
        # (a) contactar con ese nodo específico y 
        # (b) esperar a que esté listo para recibir.
        mpi_comm.send(
            obj=0,
            dest=i,
            tag=0,
        )

    # Antes de terminar, crea una barrera de sincronización MPI. 
    # Esta es la única forma correcta de sincronizar con MPI. 
    # ¿La necesitamos aquí? ¡No! Solo la estamos mostrando.
    mpi_comm.barrier()

    # We're all done!
    return 0

# Our code for ranks 1 and up
def mpi_nonroot(mpi_comm):
    # type: (MPIComm) -> int
    """Routine for the MPI workers (ranks 1 and up).

    Recibe un número de una transmisión.
    A ese número recibido, se le suma nuestro rango MPI (un entero positivo distinto de cero). 
    Mediante el proceso de recopilación, se devuelve una tupla con dos elementos: 
    * El "Identificador de CPU" MPI (normalmente un nombre de host) 
    * El número calculado (ver arriba). 
    A continuación, se inicia un bucle: Recibimos un número (un `int`) del controlador. 
    Si el número es cero, salimos del bucle. 
    De lo contrario, dividimos el número entre dos, convertimos el resultado a un int y lo enviamos al controlador. 
    Finalmente, al finalizar el bucle, sincronizamos mediante una barrera MPI.
    """

    # Obtenga nuestro rango MPI. 
    # Este es un número único, en el rango [0, MPI_size), que nos identifica 
    # en este mundo MPI.
    mpi_rank = mpi_comm.Get_rank()

    # Obtén el número que nos transmiten. 
    # `bcast()` toma un parámetro, pero como no estamos enviando, usamos `None`. 
    # NOTA: Esto se bloquea hasta que el rango cero transmite algo, pero no está 
    # sincronizando, en ese caso, el rango cero ya podría haberse movido. 
    # Y nuevamente, no hay forma de saber exactamente cómo nos llegó esto
    random_number = mpi_comm.bcast(None) # type: int

    # Comprobación de cordura: ¿realmente obtuvimos un int?
    if type(random_number) is not int:
        print('ERROR in MPI rank {}: Received a non-integer "{}" from the broadcast!'
            .format(
                mpi_rank,
                random_number,
            )
        )
        return 1

    # Nuestra respuesta es el número aleatorio + nuestro rango
    response_number = random_number + mpi_rank

    # Devolvemos nuestra respuesta 
    # `gather()` sabe que no somos la raíz, por eso regresamos.
    response = (
        MPI.Get_processor_name(),
        response_number,
    )
    mpi_comm.gather(response)

    # Recibir un mensaje unidireccional (un `int`) del controlador. 
    # Cada vez que obtenemos un entero distinto de cero, devolvemos `floor(int/2)`. 
    # Cuando obtenemos un cero, nos detenemos.

    # Mueva la recepción de mensajes a un ayudante.
    def get_message(mpi_comm):
        # type: (MPIComm) -> Union[int, None]
        message = mpi_comm.recv(
            source=0,
            tag=0,
        ) # type: int
        if type(message) is not int:
            print('ERROR in MPI rank {}: Received a non-integer message!'
                .format(
                    mpi_rank,
                )
            )
            return None
        else:
            return message

    # Start looping!
    message = get_message(mpi_comm)
    while (message is not None) and (message != 0):
        # Divide the number by 2, and send it back
        mpi_comm.send(
            obj=int(message/2),
            dest=0,
            tag=0,
        )

        # Get a new message
        message = get_message(mpi_comm)

    # Did we get an error?
    if message is None:
        return 1

    # Antes de terminar, crea una barrera de sincronización MPI. 
    # Esta es la única forma correcta de sincronizar con MPI. 
    # ¿La necesitamos aquí? ¡No! Solo la estamos mostrando..
    mpi_comm.barrier()

    # That's it!
    return 0

# Run main()
if __name__ == '__main__':
    import sys
    sys.exit(main())