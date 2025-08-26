import numpy as np
import matplotlib.pyplot as plt



f = lambda x: -(0.1 + (1-x)**2 - 0.1*np.cos(6*np.pi*(1-x))) + 2


x = np.linspace(0, 2, 50)
y = f(x)

plt.plot(x, y)


def iniciar_población(numero_individuos, longitud_cromosoma):
    return  np.random.randint(0,2,size=(numero_individuos,longitud_cromosoma))



P = iniciar_población(10,8)
print(P)