"""
Usar datos reales de la acción de Apple (AAPL) para calcular su volatilidad diaria y anualizada durante el último año.

Instrucciones:
Instala y carga la librería yfinance

Descarga los datos de cierre ajustado de AAPL del último año

Calcula los rendimientos logarítmicos diarios

Calcula la desviación estándar muestral (volatilidad diaria)

Convierte a volatilidad anualizada
"""
import yfinance as yf  
import numpy as np
import os

# 1) Datos
lista_empresas = []
tickers = ['AAPL']
lista_empresas.append (yf.download("AAPL", start="2024-01-01", end="2025-01-01")['Close']) # type: ignore
os.system("clear")



for empresa,ticker in zip(lista_empresas,tickers):
    rend = empresa.pct_change().dropna()
    rend_log = np.log(1 + rend)

    # 2) Parámetros
    dias_trading = 252
    volatilidad = np.std(rend_log, ddof=1).item()
    volatilidad_anualizada = volatilidad * np.sqrt(dias_trading)
    print(ticker)
    print(f"Volatilidad diaria: {volatilidad:.4%}")
    print(f"Volatilidad anualizada: {volatilidad_anualizada:.4%}\n")





