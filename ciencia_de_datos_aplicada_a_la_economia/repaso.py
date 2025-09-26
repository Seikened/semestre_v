"""Fórmulas simples de rendimiento financiero (versión mínima).
Pensado para listas de precios (mensuales, diarios, etc.).

Funciones:
 - rend_simple(precios)
 - rend_log(precios)
 - prom_rend_simple(precios)
 - anualizado_desde_precios(precios, periodos_por_anio=12)
 - rendimiento_real(precios, inflacion)
 - rendimiento_esperado(rends, probs)
"""
from datetime import date, timedelta
import yfinance as yf
import numpy as np
import polars as pl
import os
from colorstreak import Logger


def cargar_datos_yf(ticker: str, start: str, interval: str) -> pl.LazyFrame:
    data = yf.download(ticker, start=start, interval=interval)
    data.columns = ["_".join(col).strip() for col in data.columns.values]  # type: ignore
    data = data.reset_index()  # type: ignore
    lf = pl.LazyFrame(data)
    lf = lf.with_columns(pl.col("Date").dt.date().alias("date"))
    return lf


def closer_ticker(lf: pl.LazyFrame, ticker: str):
    return lf.select([f"Close_{ticker}"]).collect().to_series()





# 1. Rendimientos simples por periodo ES (precio final - precio inicial) / precio inicial
def rendimientos_simples(precios: np.ndarray) -> np.ndarray:
    return (precios[1:] - precios[:-1]) / precios[:-1]

# 2. Rendimientos logarítmicos por periodo (log(precio final / precio inicial))
def rendimientos_log(precios: np.ndarray) -> np.ndarray:
    return np.log(precios[1:] / precios[:-1])

# 3. Rendimiento promedio aritmético
def rendimiento_promedio(precios: np.ndarray) -> float:
    r_simple = rendimientos_simples(precios)
    return float(np.mean(r_simple))

# 4. Rendimiento anualizado
def rendimiento_anualizado(precios: np.ndarray, periodos_por_anio: int = 12) -> float:
    r_simple = rendimientos_simples(precios)
    r_acum = np.prod(1 + r_simple) - 1
    n_periodos = len(r_simple)
    return (1 + r_acum) ** (periodos_por_anio / n_periodos) - 1

# 5. Rendimiento real (ajustado por inflación)
def rendimiento_real(precios: np.ndarray, inflacion: float) -> float:
    r_simple = rendimientos_simples(precios)
    r_nominal = np.prod(1 + r_simple) - 1
    return float((1 + r_nominal) / (1 + inflacion) - 1)

# 6. Rendimiento esperado
def rendimiento_esperado(rendimientos: np.ndarray, probabilidades: np.ndarray) -> float:
    return float(np.sum(rendimientos * probabilidades))



# ============================  CARGA DE DATOS  =============================
# Parámetros
TICKERS = ["AAPL", "MSFT", "TSLA"]
PERIODOS_POR_ANIO = 12
# 3 años hacia atrás desde hoy
hoy = date.today()
fecha_inicio = hoy - timedelta(days=365 * 3 + 10)  # margen por meses incompletos

# Apple
lf_apple = cargar_datos_yf("AAPL", start=fecha_inicio.isoformat(), interval="1mo")
apple_precios = closer_ticker(lf_apple, "AAPL").to_numpy()

# Microsoft
lf_msft = cargar_datos_yf("MSFT", start=fecha_inicio.isoformat(), interval="1mo")
msft_precios = closer_ticker(lf_msft, "MSFT").to_numpy()

# Tesla
lf_tsla = cargar_datos_yf("TSLA", start=fecha_inicio.isoformat(), interval="1mo")
tsla_precios = closer_ticker(lf_tsla, "TSLA").to_numpy()



# ============================ EJECUCIÓN ============================
os.system("clear")
precios = 3


# 1 ) Rendimientos simples
print(F"\n {"="*50} RENDIMIENTOS SIMPLES {"="*50}")
apple_rend_simple = rendimientos_simples(apple_precios)
msft_rend_simple = rendimientos_simples(msft_precios)
tsla_rend_simple = rendimientos_simples(tsla_precios)
Logger.info(f"Apple : {apple_rend_simple[:precios]} | Microsoft: {msft_rend_simple[:precios]} | Tesla: {tsla_rend_simple[:precios]}")



# 2 ) Rendimientos logarítmicos
print(F"\n {"="*50} RENDIMIENTOS LOGARÍTMICOS {"="*50}")
apple_rend_log = rendimientos_log(apple_precios)
msft_rend_log = rendimientos_log(msft_precios)
tsla_rend_log = rendimientos_log(tsla_precios)
Logger.info(f"Apple : {apple_rend_log[:precios]} | Microsoft: {msft_rend_log[:precios]} | Tesla: {tsla_rend_log[:precios]}")


# 3 ) Rendimiento promedio
print(F"\n {"="*50} RENDIMIENTOS PROMEDIO {"="*50}")
apple_rend_prom = rendimiento_promedio(apple_precios)
msft_rend_prom = rendimiento_promedio(msft_precios)
tsla_rend_prom = rendimiento_promedio(tsla_precios)
Logger.info(f"Apple : {apple_rend_prom:.4f} | Microsoft: {msft_rend_prom:.4f} | Tesla: {tsla_rend_prom:.4f}")


# 4 ) Rendimiento anualizado
print(F"\n {"="*50} RENDIMIENTOS ANUALIZADOS {"="*50}")
apple_rend_anual = rendimiento_anualizado(apple_precios, PERIODOS_POR_ANIO)
msft_rend_anual = rendimiento_anualizado(msft_precios, PERIODOS_POR_ANIO)
tsla_rend_anual = rendimiento_anualizado(tsla_precios, PERIODOS_POR_ANIO)
Logger.info(f"Apple : {apple_rend_anual:.2%} | Microsoft: {msft_rend_anual:.2%} | Tesla: {tsla_rend_anual:.2%}")