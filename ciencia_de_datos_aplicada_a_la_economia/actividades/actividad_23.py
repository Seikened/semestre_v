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
    rf_anual = 0.0747   # EJEMPLO: 7.47% anual CETES 91d (reemplaza por dato vigente)
    rf_diaria = rf_anual / dias_trading

    # 3) Sharpe
    rp = rend.mean().item()
    sigma = rend.std(axis=0).item()
    sharpe_diario = (rp - rf_diaria) / sigma
    sharpe_anual = sharpe_diario * np.sqrt(dias_trading)
    
    # Aplicamos Sortino
       # Aplicamos Sortino (diario -> anualizado)
    # 4) Sortino (diario -> anual)
    exceso = rp - rf_diaria                
    downside = rend[rend < rf_diaria]  - rf_diaria 
    sigma_down_diaria = np.sqrt(np.mean(downside**2))
    sortino_diario = exceso / sigma_down_diaria 
    sortino_anual = sortino_diario * np.sqrt(252)

    print(ticker)
    print("--------- sharpe ------------")
    print(f"Rend. diario promedio: {rp:.2%}")
    print(f"Volatilidad diaria   : {sigma:.2%}")
    print(f"Sharpe diario        : {sharpe_diario:.2f}")
    print(f"Sharpe anualizado    : {sharpe_anual:.2f}\n")

    print("--------- sortino ------------")
    print(f"Exceso diario prom.  : {exceso:.2%}")
    print(f"Vol. a la baja diaria: {sigma_down_diaria:.2%}")
    print(f"Sortino diario       : {sortino_diario:.2f}")
    print(f"Sortino anualizado   : {sortino_anual:.2f}\n")   
