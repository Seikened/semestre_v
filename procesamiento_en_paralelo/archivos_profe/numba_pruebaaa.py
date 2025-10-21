#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
BENCHMARK NumPy vs Numba vs Joblib — CPU-bound & I/O-bound (estable y modular)
Autor: Fer (con cariño de Sam)
Descripción:
  - CPU-bound: kernel "grueso" de punto flotante (alto cómputo por byte).
  - I/O-bound: tareas con latencia simulada (para ver beneficio de concurrencia).
  - Comparaciones: NumPy, Numba (single/prange), Numba+joblib (threads/procs),
                   y NumPy+joblib (ref).
  - Estabilidad numérica: promote a float64 interno + clipping; sin NaNs.
  - Sin sobre-suscripción: control explícito de hilos/entornos.

Edítalo sin miedo: todos los parámetros clave están agrupados abajo.
===============================================================================
"""

# =============================== IMPORTS =====================================
import os, time, math
os.environ.setdefault("OMP_NUM_THREADS", "1")    # evita hilos ocultos de BLAS
os.environ.setdefault("MKL_NUM_THREADS", "1")   # (especialmente en NumPy)
# NUMBA_NUM_THREADS se setea más abajo en función de N_JOBS

import numpy as np
from joblib import Parallel, delayed
from numba import njit, prange

# ========================= PARÁMETROS GLOBALES ===============================
# ---- Datos y ejecución
DTYPE  = np.float32         # cambia a float64 para validación estricta
N      = 8_000_000          # tamaño del array para CPU-bound
N_JOBS = 4                  # núcleos lógicos a usar por joblib / numba
CHUNK  = 1_000_000          # tamaño de chunk para joblib (amortiza overhead)

# ---- Kernel CPU-bound
ITERS  = 8                  # sube a 16/32 para intensificar cómputo
CLIP   = 32.0               # clip estable previo a exp/cos
FASTMATH = False            # True para exprimir rendimiento (tras validar)

# ---- Suite I/O-bound (latencia simulada)
IO_TASKS     = 40           # nº de tareas (sube si tienes más cores)
IO_CHUNK_N   = 200_000      # tamaño de trabajo por tarea (cálculo ligero)
IO_SLEEP_SEC = 0.004        # "latencia" simulada por tarea (~4ms)

# Alinear hilos de Numba con joblib
os.environ.setdefault("NUMBA_NUM_THREADS", str(N_JOBS))

# ============================= UTILIDADES ====================================
def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Error relativo máximo (tolerante a magnitudes pequeñas)."""
    denom = np.maximum(np.abs(b), 1e-6)
    return float(np.max(np.abs(a - b) / denom))

def wall(op, *args, **kwargs) -> float:
    """Cronómetro helper."""
    t0 = time.perf_counter()
    op(*args, **kwargs)
    return time.perf_counter() - t0

def banner(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

# ========================== KERNEL CPU-BOUND =================================
# --- Definición matemática (idéntica entre rutas)
def f_numpy(x: np.ndarray, out: np.ndarray) -> None:
    """Baseline vectorizado con estabilidad: float64 interno + clipping."""
    xd = x.astype(np.float64, copy=False)
    t = np.sin(xd) + 0.25 * xd * xd
    for _ in range(ITERS):
        t = np.clip(t, -CLIP, CLIP)
        e = np.exp(-t)
        c = np.cos(t)
        t = ((0.1*t + 0.05)*t + 0.01)*t + e + c
    out[:] = t.astype(x.dtype, copy=False)

@njit(parallel=False, fastmath=FASTMATH, cache=True)
def fused_kernel_single(x: np.ndarray, y: np.ndarray) -> None:
    """Kernel Numba: 1 hilo, un solo paseo, misma matemática."""
    n = x.size
    for i in range(n):
        td = float(x[i])  # promover a 64-bit interno
        t = math.sin(td) + 0.25 * td * td
        for _ in range(ITERS):
            if t > CLIP: t = CLIP
            elif t < -CLIP: t = -CLIP
            e = math.exp(-t)
            c = math.cos(t)
            t = ((0.1*t + 0.05)*t + 0.01)*t + e + c
        y[i] = np.float32(t)

@njit(parallel=True, fastmath=FASTMATH, cache=True)
def fused_kernel_prange(x: np.ndarray, y: np.ndarray) -> None:
    """Kernel Numba: prange (multihilo), mismo paseo y matemática."""
    n = x.size
    for i in prange(n):
        td = float(x[i])
        t = math.sin(td) + 0.25 * td * td
        for _ in range(ITERS):
            if t > CLIP: t = CLIP
            elif t < -CLIP: t = -CLIP
            e = math.exp(-t)
            c = math.cos(t)
            t = ((0.1*t + 0.05)*t + 0.01)*t + e + c
        y[i] = np.float32(t)

# ---- Wrappers de ejecución (CPU-bound)
def run_numpy(x, out) -> float:
    return wall(f_numpy, x, out)

def run_numba_single(x, out) -> float:
    return wall(fused_kernel_single, x, out)

def run_numba_prange(x, out) -> float:
    return wall(fused_kernel_prange, x, out)

def _numba_chunk(x, y, s, e):
    fused_kernel_single(x[s:e], y[s:e])

def run_numba_joblib_threads(x, out, n_jobs=N_JOBS, chunk=CHUNK) -> float:
    tasks = [(s, min(s+chunk, x.size)) for s in range(0, x.size, chunk)]
    t0 = time.perf_counter()
    Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_numba_chunk)(x, out, s, e) for (s, e) in tasks
    )
    return time.perf_counter() - t0

def run_numba_joblib_procs(x, out, n_jobs=N_JOBS, chunk=CHUNK) -> float:
    """Procesos con retorno de chunks (evita readonly/memmap raros)."""
    tasks = [(s, min(s+chunk, x.size)) for s in range(0, x.size, chunk)]
    def _work(x, s, e):
        tmp = np.empty(e - s, dtype=x.dtype)
        fused_kernel_single(x[s:e], tmp)
        return s, e, tmp
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_work)(x, s, e) for (s, e) in tasks
    )
    for s, e, tmp in results:
        out[s:e] = tmp
    return time.perf_counter() - t0

# Para referencia: “partir” la versión NumPy
def _numpy_chunk(x, y, s, e):
    tmp = np.empty(e - s, dtype=x.dtype)
    f_numpy(x[s:e], tmp)
    y[s:e] = tmp

def run_numpy_joblib_threads(x, out, n_jobs=N_JOBS, chunk=CHUNK) -> float:
    tasks = [(s, min(s+chunk, x.size)) for s in range(0, x.size, chunk)]
    t0 = time.perf_counter()
    Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_numpy_chunk)(x, out, s, e) for (s, e) in tasks
    )
    return time.perf_counter() - t0

# ============================ SUITE I/O-BOUND =================================
# Simulamos tareas con pequeña carga numérica + latencia.
def io_task(seed: int, n: int, sleep_s: float) -> float:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n, dtype=DTYPE)
    b = rng.standard_normal(n, dtype=DTYPE)
    # mini-cómputo
    c = (a * b).sum(dtype=np.float64)
    time.sleep(sleep_s)  # “latencia” (I/O simulado)
    return float(c)

def run_io_serial(tasks=IO_TASKS, n=IO_CHUNK_N, sleep_s=IO_SLEEP_SEC) -> float:
    t0 = time.perf_counter()
    acc = 0.0
    for i in range(tasks):
        acc += io_task(i, n, sleep_s)
    _ = acc
    return time.perf_counter() - t0

def run_io_joblib_threads(tasks=IO_TASKS, n=IO_CHUNK_N, sleep_s=IO_SLEEP_SEC,
                          n_jobs=N_JOBS) -> float:
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(io_task)(i, n, sleep_s) for i in range(tasks)
    )
    _ = sum(res)
    return time.perf_counter() - t0

def run_io_joblib_procs(tasks=IO_TASKS, n=IO_CHUNK_N, sleep_s=IO_SLEEP_SEC,
                        n_jobs=N_JOBS) -> float:
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(io_task)(i, n, sleep_s) for i in range(tasks)
    )
    _ = sum(res)
    return time.perf_counter() - t0

# ================================ MAIN ========================================
if __name__ == "__main__":
    banner("CONFIG")
    print(f"DTYPE={DTYPE.__name__} | N={N} | N_JOBS={N_JOBS} | CHUNK={CHUNK} | "
          f"ITERS={ITERS} | CLIP={CLIP} | FASTMATH={FASTMATH}")
    print("THREAD LIMITS -> OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, "
          f"NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS')}")

    # -------------------------- Datos & Warm-up -------------------------------
    x = np.linspace(0, 2*np.pi, N, endpoint=False, dtype=DTYPE)
    out_np  = np.empty_like(x)
    out_nb1 = np.empty_like(x)
    out_nbP = np.empty_like(x)
    out_jbt = np.empty_like(x)
    out_jbp = np.empty_like(x)
    out_npj = np.empty_like(x)

    # Compila Numba (warm-up pequeño)
    fused_kernel_single(x[:1000], out_nb1[:1000])
    fused_kernel_prange(x[:1000], out_nbP[:1000])

    # -------------------------- CPU-bound Bench -------------------------------
    banner("CPU-BOUND — KERNEL GRUESO, UNA PASADA")
    t_np   = run_numpy(x, out_np)
    t_nb1  = run_numba_single(x, out_nb1)
    t_nbP  = run_numba_prange(x, out_nbP)
    t_jbt  = run_numba_joblib_threads(x, out_jbt)
    t_jbp  = run_numba_joblib_procs(x, out_jbp)
    t_npjt = run_numpy_joblib_threads(x, out_npj)

    # Exactitud contra baseline NumPy
    e_nb1  = rel_err(out_nb1, out_np)
    e_nbP  = rel_err(out_nbP, out_np)
    e_jbt  = rel_err(out_jbt, out_np)
    e_jbp  = rel_err(out_jbp, out_np)
    e_npjt = rel_err(out_npj, out_np)

    print(f"NumPy (vector)               : {t_np:.3f} s   err=0")
    print(f"Numba single (1 hilo)        : {t_nb1:.3f} s   err={e_nb1:.2e}")
    print(f"Numba prange (multihilo)     : {t_nbP:.3f} s   err={e_nbP:.2e}")
    print(f"Numba+joblib threading       : {t_jbt:.3f} s   err={e_jbt:.2e}")
    print(f"Numba+joblib procesos (loky) : {t_jbp:.3f} s   err={e_jbp:.2e}")
    print(f"NumPy+joblib threading (ref) : {t_npjt:.3f} s   err={e_npjt:.2e}")

    # --------------------------- I/O-bound Bench ------------------------------
    banner("I/O-BOUND — LATENCIA SIMULADA (sleep)")
    t_io_serial = run_io_serial()
    t_io_thr    = run_io_joblib_threads()
    t_io_proc   = run_io_joblib_procs()

    print(f"Serial (baseline latente)    : {t_io_serial:.3f} s")
    print(f"Joblib threading (I/O)       : {t_io_thr:.3f} s")
    print(f"Joblib procesos  (I/O)       : {t_io_proc:.3f} s")

    banner("NOTAS")
    print("* CPU-bound: espera que 'Numba prange' gane cuando ITERS es alto.")
    print("* I/O-bound: threading suele ganar por menor overhead; procesos ayudan si el trabajo por tarea es grande o CPU+I/O mixto.")
    print("* Valida exactitud con FASTMATH=False y DTYPE=float64; luego activa FASTMATH y sube ITERS para rendimiento máximo.")