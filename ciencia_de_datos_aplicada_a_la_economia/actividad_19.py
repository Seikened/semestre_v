""" "
Enunciado:
Descarga 3 tickers (p. ej., AAPL, MSFT, TSLA) de 3 años.
a) Obtén rendimientos mensuales.
b) Calcula promedio aritmético para cada activo.
c) Convierte el promedio mensual a anualizado.

"""

import yfinance as yf
import polars as pl
import matplotlib.pyplot as plt


#  Apple |  Desde que existe | Mensual
data_apple = yf.download("AAPL", start="2008-01-01", interval="1mo")
#  Microsoft |  Desde que existe | Mensual
data_microsoft = yf.download("MSFT", start="2008-01-01", interval="1mo")
# Google
data_google = yf.download("GOOGL", start="2008-01-01", interval="1mo")

# Pasamos las columnas a un formato más amigable para polars

# Apple
data_apple.columns = ["_".join(col).strip() for col in data_apple.columns.values]  # type: ignore
data_apple = data_apple.reset_index()  # type: ignore

# Microsoft
data_microsoft.columns = [
    "_".join(col).strip() for col in data_microsoft.columns.values
]  # type: ignore
data_microsoft = data_microsoft.reset_index()  # type: ignore

# Google
data_google.columns = ["_".join(col).strip() for col in data_google.columns.values]  # type: ignore
data_google = data_google.reset_index()  # type: ignore

# Dataframe de Apple
lf_apple = pl.LazyFrame(data_apple)
lf_apple = lf_apple.with_columns(pl.col("Date").dt.date().alias("date"))

# Dataframe de Microsoft
lf_microsoft = pl.LazyFrame(data_microsoft)
lf_microsoft = lf_microsoft.with_columns(pl.col("Date").dt.date().alias("date"))

# Dataframe de Google
lf_google = pl.LazyFrame(data_google)
lf_google = lf_google.with_columns(pl.col("Date").dt.date().alias("date"))


# print("\n [POLARS] Apple dataframe:")
# print(lf_apple.limit(5).collect())
# print(f"\n [POLARS] lf_apple.columns: \n{lf_apple.columns}")


serie_precios_apple = lf_apple.select("Close_AAPL").collect().to_series()

serie_precios_microsoft = lf_microsoft.select("Close_MSFT").collect().to_series()

serie_precios_google = lf_google.select("Close_GOOGL").collect().to_series()


plt.plot(serie_precios_apple, label="Precio Cierre Apple")
plt.plot(serie_precios_microsoft, label="Precio Cierre Microsoft")
plt.plot(serie_precios_google, label="Precio Cierre Google")
plt.title("Precio Cierre Apple, Microsoft y Google")
plt.xlabel("Meses")
plt.ylabel("Precio (USD)")
plt.legend()
plt.show()


# ============================ RENDIMIENTOS APPLE ============================
rendimientos_logaritmicos_apple = (
    lf_apple.select("date", "Close_AAPL")
    .rename({"Close_AAPL": "Close"})
    .with_columns(
        (pl.col("Close") / pl.col("Close").shift(1))
        .log()
        .alias("rendimiento_logaritmico")
    )
    .drop_nulls("rendimiento_logaritmico")  # eliminamos la fila inicial
)
rendimientos_logaritmicos_apple = (
    rendimientos_logaritmicos_apple.select("rendimiento_logaritmico")
    .collect()
    .to_series()
)


rendimiento_simple_apple = (
    lf_apple.select("date", "Close_AAPL")
    .rename({"Close_AAPL": "Close"})
    .with_columns(
        (pl.col("Close") / pl.col("Close").shift(1) - 1).alias("rendimiento_simple")
    )
    .drop_nulls("rendimiento_simple")  # eliminamos la fila inicial
)
rendimiento_simple_apple = (
    rendimiento_simple_apple.select("rendimiento_simple").collect().to_series()
)

# ============================ RENDIMIENTOS MICROSOFT ============================

rendimientos_logaritmicos_microsoft = (
    lf_microsoft.select("date", "Close_MSFT")
    .rename({"Close_MSFT": "Close"})
    .with_columns(
        (pl.col("Close") / pl.col("Close").shift(1))
        .log()
        .alias("rendimiento_logaritmico")
    )
    .drop_nulls("rendimiento_logaritmico")  # eliminamos la fila inicial
)
rendimientos_logaritmicos_microsoft = (
    rendimientos_logaritmicos_microsoft.select("rendimiento_logaritmico")
    .collect()
    .to_series()
)


rendimiento_simple_microsoft = (
    lf_microsoft.select("date", "Close_MSFT")
    .rename({"Close_MSFT": "Close"})
    .with_columns(
        (pl.col("Close") / pl.col("Close").shift(1) - 1).alias("rendimiento_simple")
    )
    .drop_nulls("rendimiento_simple")  # eliminamos la fila inicial
)
rendimiento_simple_microsoft = (
    rendimiento_simple_microsoft.select("rendimiento_simple").collect().to_series()
)


# ============================ RENDIMIENTOS GOOGLE ============================
rendimientos_logaritmicos_google = (
    lf_google.select("date", "Close_GOOGL")
    .rename({"Close_GOOGL": "Close"})
    .with_columns(
        (pl.col("Close") / pl.col("Close").shift(1))
        .log()
        .alias("rendimiento_logaritmico")
    )
    .drop_nulls("rendimiento_logaritmico")  # eliminamos la fila inicial
)
rendimientos_logaritmicos_google = (
    rendimientos_logaritmicos_google.select("rendimiento_logaritmico")
    .collect()
    .to_series()
)


rendimiento_simple_google = (
    lf_google.select("date", "Close_GOOGL")
    .rename({"Close_GOOGL": "Close"})
    .with_columns(
        (pl.col("Close") / pl.col("Close").shift(1) - 1).alias("rendimiento_simple")
    )
    .drop_nulls("rendimiento_simple")  # eliminamos la fila inicial
)
rendimiento_simple_google = (
    rendimiento_simple_google.select("rendimiento_simple").collect().to_series()
)


import os

os.system("clear")


# ============================ RESULTADOS ============================
print("\n Rendimientos logarítmicos Apple:")
apple_promedio_logaritmico = rendimientos_logaritmicos_apple.mean()
apple_promedio_logaritmico_anualizado = ((1 + apple_promedio_logaritmico) ** 12) - 1
print(f"Promedio logarítmico mensual: {apple_promedio_logaritmico * 100:.2f}%")
print(f"Promedio logarítmico anualizado: {apple_promedio_logaritmico_anualizado * 100:.2f}%")

print("\n Rendimientos simples Apple:")
apple_promedio_simple = rendimiento_simple_apple.mean()
apple_promedio_simple_anualizado = ((1 + apple_promedio_simple) ** 12) - 1
print(f"Promedio simple mensual: {apple_promedio_simple * 100:.2f}%")
print(f"Promedio simple anualizado: {apple_promedio_simple_anualizado * 100:.2f}%")


print("\n Rendimientos logarítmicos Microsoft:")
microsoft_promedio_logaritmico = rendimientos_logaritmicos_microsoft.mean()
microsoft_promedio_logaritmico_anualizado = ((1 + microsoft_promedio_logaritmico) ** 12) - 1
print(f"Promedio logarítmico mensual: {microsoft_promedio_logaritmico * 100:.2f}%")
print(f"Promedio logarítmico anualizado: {microsoft_promedio_logaritmico_anualizado * 100:.2f}%")

print("\n Rendimientos simples Microsoft:")
microsoft_promedio_simple = rendimiento_simple_microsoft.mean()
microsoft_promedio_simple_anualizado = ((1 + microsoft_promedio_simple) ** 12) - 1
print(f"Promedio simple mensual: {microsoft_promedio_simple * 100:.2f}%")
print(f"Promedio simple anualizado: {microsoft_promedio_simple_anualizado * 100:.2f}%")


print("\n Rendimientos logarítmicos Google:")
google_promedio_logaritmico = rendimientos_logaritmicos_google.mean()
google_promedio_logaritmico_anualizado = ((1 + google_promedio_logaritmico) ** 12) - 1
print(f"Promedio logarítmico mensual: {google_promedio_logaritmico * 100:.2f}%")
print(f"Promedio logarítmico anualizado: {google_promedio_logaritmico_anualizado * 100:.2f}%")

print("\n Rendimientos simples Google:")
google_promedio_simple = rendimiento_simple_google.mean()
google_promedio_simple_anualizado = ((1 + google_promedio_simple) ** 12) - 1
print(f"Promedio simple mensual: {google_promedio_simple * 100:.2f}%")
print(f"Promedio simple anualizado: {google_promedio_simple_anualizado * 100:.2f}%")


