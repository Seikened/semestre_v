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




def cargar_datos_yf(ticker: str, start: str, interval: str) -> pl.LazyFrame:
    data = yf.download(ticker, start=start, interval=interval)
    data.columns = ["_".join(col).strip() for col in data.columns.values]  # type: ignore
    data = data.reset_index()  # type: ignore
    lf = pl.LazyFrame(data)
    lf = lf.with_columns(pl.col("Date").dt.date().alias("date"))
    return lf


def closer_ticker(lf: pl.LazyFrame, ticker: str):
    return lf.select([f"Close_{ticker}"]).collect().to_series()



#  Apple |  Desde que existe | Mensual
lf_apple = cargar_datos_yf("AAPL", start="2008-01-01", interval="1mo")
#  Microsoft |  Desde que existe | Mensual
lf_microsoft = cargar_datos_yf("MSFT", start="2008-01-01", interval="1mo")
# Google
lf_google = cargar_datos_yf("GOOGL", start="2008-01-01", interval="1mo")


serie_precios_apple = closer_ticker(lf_apple, "AAPL")

serie_precios_microsoft = closer_ticker(lf_microsoft, "MSFT")

serie_precios_google = closer_ticker(lf_google, "GOOGL")


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


