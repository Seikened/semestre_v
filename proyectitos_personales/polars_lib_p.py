import polars as pl
import yfinance as yf
import numpy as np
import seaborn as sns
from datetime import date, timedelta
import time


#  Amazon | FECHA: 15 de mayo de 1997 hasta hoy
data_amazon = yf.download("AMZN", start="1997-05-15")


# Google | 19 de agosto de 2004 hasta hoy
data_google = yf.download("GOOGL", start="2004-08-19")

print("\n [PANDAS] Amazon data:")
print(data_amazon.head()) # type: ignore
print(f"\n [PANDAS] data_amazon.columns: \n{data_amazon.columns}") # type: ignore

print("\n [PANDAS] Google data:")
print(data_google.head()) # type: ignore
print(f"\n [PANDAS] data_google.columns: \n{data_google.columns}") # type: ignore


data_amazon.columns = ["_".join(col).strip() for col in data_amazon.columns.values] # type: ignore
data_amazon = data_amazon.reset_index() # type: ignore

data_google.columns = ["_".join(col).strip() for col in data_google.columns.values] # type: ignore
data_google = data_google.reset_index() # type: ignore

# Dataframe de Amazon
df_amazon = pl.DataFrame(data_amazon).with_columns(
    pl.col("Date").dt.date().alias("Date")
)

# Dataframe de Google
df_google = pl.DataFrame(data_google).with_columns(
    pl.col("Date").dt.date().alias("Date")
)

print("\n [PANDAS] Amazon dataframe:")
print(df_amazon.head())
print(f"\n [PANDAS] df_amazon.columns: \n{df_amazon.columns}")

print("\n [PANDAS] Google dataframe:")
print(df_google.head())
print(f"\n [PANDAS] df_google.columns: \n{df_google.columns}")