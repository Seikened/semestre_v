"""
FERNANDO LEON FRANCO
EXAMEN
PREGUNTAS DEL EXAMEN
"""


import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl


os.system("clear")



# PREGUNTA 5
# Con los rendimientos [0.08, 0.09, 0.07, 0.12] y una tasa libre de riesgo rf = 0.03, calcula el Ratio de Sharpe.
rendimientos = np.array([0.08, 0.09, 0.07, 0.12])
tasa_libre_riesgo = 0.03
primedio_rendimientos = np.mean(rendimientos)
desviacion_estandar = np.std(rendimientos, ddof=1)
ratio_sharpe = (primedio_rendimientos - tasa_libre_riesgo) / desviacion_estandar
print("=" * 70)
print(f"[RESPUESTA PREGUNTA 5] Ratio de Sharpe: {ratio_sharpe:.4f}")





# PREGUNTA 7
# Escribe un programa en Python que grafique los siguientes rendimientos usando matplotlib: [0.02, 0.03, -0.01, 0.015, 0.04].
x = [0.02, 0.03, -0.01, 0.015, 0.04]
y = range(len(x))
plt.plot(y, x, marker='o')
plt.title("Rendimientos")
plt.xlabel("Períodos")
plt.ylabel("Rendimiento")
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
#plt.show()



# PREGUNTA 14 
# Usa numpy para calcular la desviación estándar de los siguientes rendimientos: [0.05, 0.02, -0.01, 0.03, 0.04].
rendimientos = np.array([0.05, 0.02, -0.01, 0.03, 0.04])
desviacion_estandar = np.std(rendimientos, ddof=1)
print("=" * 70)
print(f"[RESPUESTA PREGUNTA 14] Desviación estándar: {desviacion_estandar:.2f}")



# PREGUNTA 16
# Usa pandas (y polars) para convertir la lista [120, 125, 130, 128, 132] en una serie y calcula los rendimientos logarítmicos.
def rendimientos_log(precios: np.ndarray) -> np.ndarray:
    return np.log(precios[1:] / precios[:-1])

precios = [120, 125, 130, 128, 132]
datos_empresa ={
    "precios": precios,
}
dataframe_pandas = pd.DataFrame(datos_empresa)
precios_pandas = dataframe_pandas['precios'].to_numpy()

dataframe_polars = pl.DataFrame(datos_empresa)
precios_polars = dataframe_polars['precios'].to_numpy()

rendimientos_log_pandas = rendimientos_log(precios_pandas)
rendimientos_log_polars = rendimientos_log(precios_polars)

print("=" * 70)
print(f"[RESPUESTA PREGUNTA 16] Rendimientos logarítmicos (Pandas):\n{rendimientos_log_pandas}")
print(f"[RESPUESTA PREGUNTA 16] Rendimientos logarítmicos (Polars):\n{rendimientos_log_polars}")




# pregunta 17
# Con la lista precios = [100, 105, 110, 108, 112], calcula los rendimientos simples entre cada par de días consecutivos sin usar pandas.
def rendimientos_simples(precios: np.ndarray) -> np.ndarray:
    return (precios[1:] - precios[:-1]) / precios[:-1]

precios = np.array([100, 105, 110, 108, 112])
rendimientos_simples_resultado = rendimientos_simples(precios)

print("=" * 70)
print(f"[RESPUESTA PREGUNTA 17] Rendimientos simples:\n{rendimientos_simples_resultado}")



# pregunta 18
# Escribe código en Python que calcule el rendimiento anualizado si el rendimiento mensual es de 5%.
def rendimiento_anualizado(precios: np.ndarray, periodos_por_anio: int) -> float:
    r_simple = rendimientos_simples(precios)
    r_acum = np.prod(1 + r_simple) - 1
    n_periodos = len(r_simple)
    return (1 + r_acum) ** (periodos_por_anio / n_periodos) - 1

precios_con_rendimiento_cinco_porciento = [ x* 1.05 for x in range(100, 200, 5)]
precios_con_rendimiento = np.array(precios_con_rendimiento_cinco_porciento)

periodo = 12
rendimientos_anual = rendimiento_anualizado(precios_con_rendimiento,periodo)

print("=" * 70)
print(f"[RESPUESTA PREGUNTA 18] Rendimiento anualizado:\n{rendimientos_anual}")   



# PREGUNTA 21
# Con la lista [0.05, -0.02, 0.00, 0.07, -0.01], clasifica cada rendimiento como positivo, negativo o neutro. 
rendimientos = [0.05, -0.02, 0.00, 0.07, -0.01]
print("=" * 70)
for precio in rendimientos:
    if precio > 0:
        clasificacion = "✅ positivo"
    elif precio < 0:
        clasificacion = "❌ negativo"
    else:
        clasificacion = "⚪ neutro"

    print(f"[RESPUESTA PREGUNTA 21] Rendimiento {precio} es {clasificacion}")
    
    
# PREGUNTA 25
# Escribe una función en Python llamada interes_compuesto que reciba capital, tasa y tiempo, y retorne el monto final.

def interes_compuesto(capital: float, tasa: float, tiempo: int) -> float:
    return capital * (1 + tasa) ** tiempo

capital = 1000
tasa = 0.05
tiempo = 10

resultado = interes_compuesto(capital, tasa, tiempo)

print("=" * 70)
print(f"[RESPUESTA PREGUNTA 25] Capital inicial: {capital}| Tasa de crecimiento: {tasa} | Tiempo: {tiempo} | Monto final: {resultado:.2f}")



# PREGUNTA 28
# Escribe un programa en Python que calcule el rendimiento simple entre un precio inicial de 100 y un precio final de 115.
precio_inicial = 100
precio_final = 115
rendimientos = np.linspace(precio_inicial, precio_final, 10)
rendimientos = np.array(rendimientos)

rendimientos_simples_finales = rendimientos_simples(rendimientos)
print("=" * 70)
print(f"[RESPUESTA PREGUNTA 28] Rendimiento simple entre {precio_inicial} y {precio_final}: {rendimientos_simples_finales}")

