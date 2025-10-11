import numpy as np

vol_diaria = 0.017
vol_anualizada = vol_diaria * np.sqrt(252)

print(f"Volatilidad anualizada: {vol_anualizada:.2%}")