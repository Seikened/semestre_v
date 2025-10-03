import yfinance as yf
import numpy as np
import os

# 1) Datos
# America movil en 2024 (MX)
lista_empresas = []
tickers = ['walmex','gmexico']
lista_empresas.append (yf.download("WALMEX.MX", start="2024-01-01", end="2025-01-01")['Close']) # type: ignore
lista_empresas.append(yf.download("GMEXICOB.MX", start="2024-01-01", end="2025-01-01")['Close'])# type: ignore
os.system("clear")



for empresa,ticker in zip(lista_empresas,tickers):
    rend = empresa.pct_change().dropna()

    # 2) Parámetros
    dias_trading = 252
    rf_anual = 0.0747   # EJEMPLO: 7.47% anual CETES 28d (reemplaza por dato vigente)
    rf_diaria = rf_anual / dias_trading

    # 3) Métricas (como escalares, sin warnings)
    rp = rend.mean().item()
    sigma = rend.std(axis=0).item()


    sharpe_diario = (rp - rf_diaria) / sigma
    sharpe_anual = sharpe_diario * np.sqrt(dias_trading)
    print(ticker)
    print(f"Rend. diario promedio: {rp:.4%}")
    print(f"Volatilidad diaria   : {sigma:.4%}")
    print(f"Sharpe diario        : {sharpe_diario:.2f}")
    print(f"Sharpe anualizado    : {sharpe_anual:.2f}\n")


