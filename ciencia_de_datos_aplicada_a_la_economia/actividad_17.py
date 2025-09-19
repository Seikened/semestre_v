
"""

Enunciado:
Descarga el precio ajustado de un ticker (p. ej., AAPL) en el último año. Calcula:

Rendimiento simple diario, 

Rendimiento logarítmico diario.


"""
from joblib import Logger
import yfinance as yf
import polars as pl
import matplotlib.pyplot as plt


#  Apple |  Desde que existe
data_apple = yf.download("AAPL", start="2008-01-01")

print("\n [PANDAS] Apple data:")
print(data_apple.head()) # type: ignore
print(f"\n [PANDAS] data_apple.columns: \n{data_apple.columns}") # type: ignore


data_apple.columns = ["_".join(col).strip() for col in data_apple.columns.values] # type: ignore
data_apple = data_apple.reset_index() # type: ignore

# Dataframe de Apple
lf_apple = pl.LazyFrame(data_apple)

lf_apple = lf_apple.with_columns(
    pl.col("Date").dt.date().alias("Date")
)



print("\n [POLARS] Apple dataframe:")
print(lf_apple.limit(5).collect())
print(f"\n [POLARS] lf_apple.columns: \n{lf_apple.columns}")



serie_precios = (
    lf_apple
    .select('Close_AAPL')
    .collect()
    .to_series()
)

plt.plot(serie_precios, label="Precio Cierre Apple")
plt.title("Precio Cierre Apple")
plt.xlabel("Días")
plt.ylabel("Precio (USD)")
plt.legend()
plt.show()




rendimientos_logartimicos = (
    lf_apple
    .select('Date', 'Close_AAPL')
    .rename({'Close_AAPL': 'Close'})
    .with_columns(
        (pl.col('Close') / pl.col('Close').shift(1))
        .log()
        .alias('rendimiento_logaritmico')
    )
    .drop_nulls("rendimiento_logaritmico")  # eliminamos la fila inicial
)

print(f"Rendimiento logaritmico {rendimientos_logartimicos.collect()}")


rendimiento_simple = (
    lf_apple
    .select('Date', 'Close_AAPL')
    .rename({'Close_AAPL': 'Close'})
    .with_columns(
        (pl.col('Close') / pl.col('Close').shift(1) - 1)
        .alias('rendimiento_simple')
    )
    .drop_nulls("rendimiento_simple")  # eliminamos la fila inicial
)

print(f"Rendimiento simple {rendimiento_simple.collect()}")




rendimientos_logartimicos = (
    rendimientos_logartimicos
    .select('rendimiento_logaritmico')
    .collect()
    .to_series()
)

# for r in rendimientos_logartimicos[10:15]:
#     print(f"Rendimiento logarítmico diario: {r:.2f}")

rendimientos_simple = (
    rendimiento_simple
    .select('rendimiento_simple')
    .collect()
    .to_series()
)

# for r in rendimientos_simple[10:15]:
#     print(f"Rendimiento simple diario: {r:.2f}")



# Rendimiento diario por los dos métodos
plt.plot(rendimientos_logartimicos, label="Rendimiento logarítmico diario")
plt.plot(rendimientos_simple, label="Rendimiento simple diario")
plt.title("Rendimientos diarios de Apple")
plt.xlabel("Días")
plt.ylabel("Rendimiento")
plt.legend()
plt.show()  


