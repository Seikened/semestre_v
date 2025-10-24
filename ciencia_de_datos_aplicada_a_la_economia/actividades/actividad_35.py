# =========================
# Ejercicio 2 - Umbral de probabilidad en Regresión Logística
# Autor: Fernando Leon Franco
# =========================

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 0 Descargar datos y preparar X, y (para independencia del código anterior)
btc = yf.download("BTC-USD", start="2023-01-01", end="2025-01-01")
btc["Rendimiento"] = np.log(btc["Close"] / btc["Close"].shift(1))
btc["Rend_prev"] = btc["Rendimiento"].shift(1)
btc["Sube"] = np.where(btc["Rendimiento"] > 0, 1, 0)
btc = btc.dropna()
X = btc[["Rend_prev", "Volume"]]
y = btc["Sube"]

# 1 Escalar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2 Reentrenar el modelo
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
modelo = LogisticRegression(class_weight='balanced', max_iter=500)
modelo.fit(X_train, y_train)

# 3 Probabilidades de subida
prob = modelo.predict_proba(X_test)[:, 1]

# 4 Cambiar umbral
umbral = 0.6  # ajusta entre 0.4 y 0.7
y_pred = (prob >= umbral).astype(int)

# 5 Evaluar
print("Umbral:", umbral)
print("Precisión:", round(accuracy_score(y_test, y_pred), 3))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# 6 Gráfica de probabilidades
plt.figure(figsize=(10, 5))
plt.plot(prob, label="Probabilidad de subida", color="red")
plt.axhline(y=umbral, color="gray", linestyle="--")
plt.title(f"Probabilidad de subida vs Umbral ({umbral})")
plt.legend()
plt.show()

# =========================
# Análisis rápido
# =========================
print("\n================= ANÁLISIS =================")
print(f"1️⃣  Umbral usado: {umbral}")
print("👉  Si el umbral sube (0.7), el modelo exige más certeza antes de decir 'sube',")
print("    lo que reduce falsos positivos pero puede aumentar falsos negativos.")

print("\n2️⃣  Si bajas el umbral (0.4), el modelo predice más subidas,")
print("    pero se arriesga a fallar más veces en bajadas.")

print("\n3️⃣  La línea gris representa el límite de decisión en la gráfica:")
print("    todo punto rojo arriba de esa línea es clasificado como 'sube' (1).")

print("\n4️⃣  Para reducir riesgo de inversión, conviene un umbral más alto (≈0.6–0.7).")
print("============================================")