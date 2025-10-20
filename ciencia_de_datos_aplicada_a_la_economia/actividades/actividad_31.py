"""
Objetivo:
Transformar los precios en rendimientos logarítmicos y predecir el rendimiento del día siguiente.

Actividad en Python:
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf

btc = yf.download("BTC-USD", start="2024-01-01", end="2025-01-01")
btc["Close_prev"] = btc["Close"].shift(1)
btc = btc.dropna()

btc["Rendimiento"] = np.log(btc["Close"] / btc["Close"].shift(1))
btc["Rend_prev"] = btc["Rendimiento"].shift(1)
btc = btc.dropna()

X = btc[["Rend_prev"]]
y = btc["Rendimiento"]

modelo = LinearRegression()
modelo.fit(X, y)

print("Intercepto:", modelo.intercept_)
print("Pendiente:", modelo.coef_[0])

print("Análisis:")

if abs(modelo.coef_[0]) < 0.1:
    print("La pendiente es cercana a 0, lo que implica que el rendimiento es difícil de predecir.")
else:
    print("La pendiente es significativa, lo que sugiere que el rendimiento puede ser predecible.")

print("Diferencias entre predecir precios y rendimientos:")
print("1. Los precios son valores absolutos, mientras que los rendimientos son tasas de cambio.")
print("2. Los rendimientos logarítmicos tienden a ser más estables y normalmente distribuidos.")