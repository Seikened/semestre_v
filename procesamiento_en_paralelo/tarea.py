from __future__ import annotations
from multiprocessing import Process, Lock, Event  # , current_process, Pipe
from dataclasses import dataclass
import numpy as np
import os
import time

MIN_STOCK = 10
MAX_STOCK = 50


@dataclass
class Producto:
    id: str
    nombre: str
    precio: float
    cantidad: int


class InventarioClass:
    def __init__(self, items, lock, restock_lock: Lock | None = None):
        self._inventario = items
        self._lock = lock
        self._restock_lock: Lock | None = restock_lock

    def añadir_producto(self, producto):
        with self._lock:
            self._inventario.append(producto)
            return True

    def verificar_stock(self):
        with self._lock:
            # devolvemos una copia “real” para imprimir sin bloquear
            return list(self._inventario)

    def tamaño_stock(self) -> int:
        with self._lock:
            return len(self._inventario)

    def sacar_producto(self, cantidad_retirar: int):
        """Retira hasta `cantidad_retirar` productos de forma FIFO y devuelve:
        (lista_productos, total_antes, total_restante) en una sola sección crítica.
        """
        with self._lock:
            total_antes = len(self._inventario)
            if cantidad_retirar <= 0 or total_antes == 0:
                return [], total_antes, total_antes
            n = min(cantidad_retirar, total_antes)
            productos = self._inventario[:n]
            del self._inventario[:n]
            restante = len(self._inventario)
            return productos, total_antes, restante


def maquila(id) -> Producto:
    producto = Producto(
        id=str(id),
        nombre=f"|{id}|-producto",
        precio=np.random.uniform(10, 100),
        cantidad=1,
    )

    return producto


def productor_logica(inventario):
    # Reabastecimiento idempotente: sólo uno entra y recalcula dentro del lock
    with inventario._restock_lock:
        curr = inventario.tamaño_stock()
        print(f"[ 🏭 PRODUCTOR]: Inventario actual {curr}")
        if curr < MIN_STOCK:
            lote = np.random.randint(1, 11)
            print(
                f"[ 🏭 PRODUCTOR]: Stock < {MIN_STOCK}, reabasteciendo 📦 | lote {lote} ..."
            )
            missing = MAX_STOCK - curr
            for i in range(missing):
                producto = maquila(f"{lote}-{i}")
                inventario.añadir_producto(producto)
            print(
                f"[ 🏭 PRODUCTOR]: Inventario reabastecido ✅ | Stock total {inventario.tamaño_stock()}"
            )


def productor_worker(inventario, restock_evt: Event, stop_evt: Event):
    while not stop_evt.is_set():
        # Espera señal (hasta 0.2s) o verifica bajo inventario de forma periódica
        señalado = restock_evt.wait(timeout=0.2)
        if señalado or inventario.tamaño_stock() < MIN_STOCK:
            productor_logica(inventario)
            restock_evt.clear()


def consumidor_logica(
    sucursal, inventario, restock_evt: Event | None = None
):
    # Hasta 2 intentos: intentamos comprar y, si no hay, pedimos reabasto y esperamos un momento.
    for intento in range(2):
        cantidad_a_comprar = np.random.randint(1, 6)
        productos, antes, restante = inventario.sacar_producto(cantidad_a_comprar)

        if productos:
            print(
                f"[ 👤 CONSUMIDOR de {sucursal}]: Se compró 🤑 {len(productos)} | "
                f"Inventario antes {antes} → después {restante}"
            )
            # Señal temprana si quedamos por debajo del mínimo
            if restock_evt is not None and restante < MIN_STOCK:
                restock_evt.set()
            return
        else:
            print(
                f"[ 👤 CONSUMIDOR de {sucursal}]: No hay productos disponibles ❌ | Esperando... ⏳ "
            )
            if restock_evt is not None:
                # Pedimos reabasto y esperamos un poco antes de reintentar
                restock_evt.set()
                restock_evt.wait(timeout=1.0)
    # Si llegamos aquí, tampoco se pudo en el reintento corto (dejamos que la siguiente hora lo intente de nuevo)


def trabajo_consumidor(
    sucursal, inventario_cedis, evento_reabasto: Event | None = None
):
    proceso = Process(
        target=consumidor_logica,
        args=(sucursal, inventario_cedis, evento_reabasto),
    )
    proceso.start()
    proceso.join()


def sucursal(
    nombre,
    inventario_cedis,
    horas_laborales: int = 8,
    evento_reabasto: Event | None = None,
):
    id_sucursal = os.getpid()
    nombre = f"{nombre.upper()}-{id_sucursal}"
    print(f"Sucursal en {nombre} iniciada | proceso padre: {os.getppid()}")
    for hora in range(horas_laborales):
        print(f"--- Sucursal de {nombre} | Hora laboral {hora + 1} ---")
        trabajo_consumidor(nombre, inventario_cedis, evento_reabasto)


if __name__ == "__main__":
    print(f"Mi proceso principal es {os.getpid()}")

    # Contexto y recursos compartidos (multiprocessing seguro con 'spawn' en macOS)
    import multiprocessing as mp

    contexto = mp.get_context("spawn")
    administrador = contexto.Manager()

    inventario_lista = administrador.list()  # lista compartida real (productos)
    bloqueo_inventario = administrador.Lock()  # lock para operaciones de inventario
    bloqueo_reabasto = administrador.Lock()  # lock exclusivo para reabastecer

    evento_reabasto = contexto.Event()  # señal para pedir reabasto
    evento_fin = contexto.Event()  # señal para finalizar productor

    # Construir CEDIS (inventario compartido)
    inventario_cedis = InventarioClass(
        inventario_lista, bloqueo_inventario, bloqueo_reabasto
    )

    # Semilla de productos iniciales
    for i in range(5):
        inventario_cedis.añadir_producto(maquila(i))

    # Iniciar productor global (único)
    proceso_productor = Process(
        target=productor_worker, args=(inventario_cedis, evento_reabasto, evento_fin)
    )
    proceso_productor.start()

    # Lanzar sucursales en paralelo (sólo consumidores)
    sucursales = []
    for nombre_sucursal in ("CDMX 🌮", "LEON 🦜", "GUADALAJARA 🐦"):
        p = Process(
            target=sucursal,
            args=(nombre_sucursal, inventario_cedis),
            kwargs=dict(horas_laborales=8, evento_reabasto=evento_reabasto),
        )
        p.start()
        sucursales.append(p)

    # Esperar fin de jornada de todas las sucursales
    for p in sucursales:
        p.join()

    # Apagar productor global
    evento_fin.set()
    proceso_productor.join()
