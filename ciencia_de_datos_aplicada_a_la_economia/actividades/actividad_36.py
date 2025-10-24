# =========================
# Ejercicio 3 - SVR vs Regresión Lineal
# Autor: Fernando Leon Franco
# =========================

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1 Descargar precios de BTC
btc = yf.download("BTC-USD", start="2024-01-01", end="2025-01-01")

# 2 Calcular variables
btc["Rendimiento"] = np.log(btc["Close"] / btc["Close"].shift(1))
btc["MA7"] = btc["Close"].rolling(7).mean()
btc["Volatilidad"] = btc["Rendimiento"].rolling(7).std()
btc = btc.dropna()

# 3 Variables predictoras y objetivo
X = btc[["MA7", "Volatilidad"]]
y = btc["Close"]

# 4 Escalado
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# 5 Modelos
svr_linear = SVR(kernel='linear', C=100)
svr_linear.fit(X_scaled, y_scaled)

lr = LinearRegression()
lr.fit(X_scaled, y_scaled)

# 6 Predicciones
y_pred_svr = scaler_y.inverse_transform(svr_linear.predict(X_scaled).reshape(-1, 1))
y_pred_lr = scaler_y.inverse_transform(lr.predict(X_scaled).reshape(-1, 1))

# 7 Evaluación
r2_svr = r2_score(y, y_pred_svr)
r2_lr = r2_score(y, y_pred_lr)

print("SVR lineal - R²:", round(r2_svr, 3))
print("Regresión lineal - R²:", round(r2_lr, 3))

# 8 Gráfico comparativo
plt.figure(figsize=(12, 5))
plt.plot(btc.index, y, label="Precio real", color="black")
plt.plot(btc.index, y_pred_svr, label="SVR lineal", color="blue")
plt.plot(btc.index, y_pred_lr, label="Regresión lineal", color="red", linestyle="--")
plt.title("Comparación SVR vs Regresión Lineal")
plt.legend()
plt.show()

# =========================
# Análisis simple
# =========================
print("\n================= ANÁLISIS =================")
if r2_svr > r2_lr:
    print("1️⃣  El modelo SVR tiene mejor R² que la regresión lineal.")
else:
    print("1️⃣  La regresión lineal obtuvo mejor R² que el SVR.")

print(f"👉  R² SVR: {round(r2_svr, 3)} | R² Lineal: {round(r2_lr, 3)}")

print("\n2️⃣  El SVR usa un margen de error (epsilon) y puede ajustarse mejor a fluctuaciones,")
print("    por eso suele seguir mejor los cambios bruscos del precio.")

print("\n3️⃣  El escalado es importante porque SVR depende de distancias entre puntos,")
print("    y si las variables tienen distintas escalas (ej. Volumen vs Precio),")
print("    una puede dominar la otra y arruinar el ajuste.")

print("============================================")