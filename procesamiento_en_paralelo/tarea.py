from multiprocessing import Process, Pool, Queue, Lock  # , current_process, Pipe
from dataclasses import dataclass
import numpy as np
import os
import time


@dataclass
class Producto:
    id: str
    nombre: str
    precio: float
    cantidad: int


class InventarioClass:
    def __init__(self, items, lock):
        self._inventario = items  # manager.list()
        self._lock = lock  # manager.Lock()

    def añadir_producto(self, producto):
        with self._lock:
            self._inventario.append(producto)
            return True

    def verificar_stock(self):
        with self._lock:
            # devolvemos una copia “real” para imprimir sin bloquear
            return list(self._inventario)

    def sacar_producto(self, cantidad_retirar: int):
        with self._lock:
            if cantidad_retirar <= 0 or len(self._inventario) == 0:
                return None
            n = min(cantidad_retirar, len(self._inventario))

            # pop(0) para simular FIFO (First In, First Out) simple
            productos = [self._inventario.pop(0) for _ in range(n)]
            return productos


def maquila(id) -> Producto:
    producto = Producto(
        id=str(id),
        nombre=f"|{id}|-producto",
        precio=np.random.uniform(10, 100),
        cantidad=1,
    )

    return producto


def productor_logica(inventario):
    lote = np.random.randint(1, 11)
    min_stock = 10
    max_stock = 50

    cantidad_inventario = inventario.verificar_stock()
    print(f"[ 🏭 PRODUCTOR]: Inventario actual {len(cantidad_inventario)}")
    if len(cantidad_inventario) < min_stock:
        print(
            f"[ 🏭 PRODUCTOR]: Inventario por debajo de {min_stock}, reabasteciendo 📦 | lote {lote} ..."
        )
        cantidad_a_rellenar = max_stock - len(cantidad_inventario)
        for i in range(cantidad_a_rellenar):
            id = f"{lote}-{i}"
            producto = maquila(id)
            inventario.añadir_producto(producto)
        print("[ 🏭 PRODUCTOR]: Inventario reabastecido ✅")


def consumidor_logica(inventario):
    cantidad_a_comprar = np.random.randint(1, 6)
    productos = inventario.sacar_producto(cantidad_a_comprar)
    if productos:
        print(f"[ 👤 CONSUMIDOR]: Se compró 🤑 {len(productos)}")
    else:
        print("[ 👤 CONSUMIDOR]: No hay productos disponibles ❌ | Esperando... ⏳ ")


def proceso_productor(inventario):
    p = Process(target=productor_logica, args=(inventario,))
    return p


def proceso_consumidor(inventario):
    p = Process(target=consumidor_logica, args=(inventario,))
    return p


def trabajo_productor(inventario_global, tasa_productor: int = 1):
    productor_entidad = proceso_productor(inventario_global)
    productor_entidad.start()
    productor_entidad.join()


def trabajo_consumidor(inventario_global, tasa_consumidor: int = 1):
    consumidor_entidad = proceso_consumidor(inventario_global)
    consumidor_entidad.start()
    consumidor_entidad.join()



def sucursal(
    nombre,
    CEDIS,
    tasa_productor: int = 1,
    tasa_consumidor: int = 1,
    horas_laborales: int = 8,
):
    id_sucursal = os.getpid()
    nombre = f"{nombre.upper()}-{id_sucursal}"
    print(f"Sucursal en {nombre} iniciada | proceso padre: {os.getppid()}")
    for hora in range(horas_laborales):
        print(f"--- Sucursal de {nombre} | Hora laboral {hora + 1} ---")
        trabajo_consumidor(CEDIS, tasa_consumidor)


if __name__ == "__main__":
    print(f"Mi proceso principal es {os.getpid()}")

    # Contexto y manager (spawn es el default en macOS; bien para multiproceso seguro)
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    items = manager.list()  # lista compartida real
    lock = manager.Lock()  # lock compartido real

    CEDIS = InventarioClass(items, lock)

    # Inicializar el inventario
    for i in range(5):
        producto = maquila(i)
        CEDIS.añadir_producto(producto)

    for dias in range(1):
        print(f"\n {10*'='} Día de trabajo {dias + 1} {10*'='} \n")
        # Configurar las tasas de producción y consumo

        tasa_productor = np.random.randint(1, 4)  # productos por hora
        encargado_inventarios_cedis = trabajo_productor(CEDIS, tasa_productor)

        tasa_consumidor = np.random.randint(1, 4)  # productos por hora

        # Iniciar sucursales
        sucursales = []
        for nombre in ("CDMX", "LEON", "GUADALAJARA"):
            p = Process(
                target=sucursal,
                args=(nombre, CEDIS),
                kwargs=dict(
                    tasa_productor=tasa_productor,
                    tasa_consumidor=tasa_consumidor,
                    horas_laborales=8,
                ),
            )
            p.start()
            sucursales.append(p)

        # Esperar a que todas las sucursales terminen su jornada
        for p in sucursales:
            p.join()
