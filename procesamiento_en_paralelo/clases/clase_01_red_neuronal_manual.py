import numpy as np
import random
import time

class RedNeuronalBackpropagation:
    def __init__(self, capas, tasa_aprendizaje: float = 0.1) -> None:
        self.capas = capas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.num_capas = len(capas)
        
        # Inicializar pesos y biases
        self.pesos = []
        self.sesgos = []
        
        for i in range(self.num_capas - 1):
            peso = np.random.rand(self.capas[i+1], self.capas[i]) * 0.1
            self.pesos.append(peso)

            sesgo = np.zeros((capas[i+1], 1))
            self.sesgos.append(sesgo)
            
    
    def funcion_activacion(self,x, derivada = False):    
        """ 
        Función de activación a implementar es la sigmoide (su derivada también)
        Args:
            x (np.array): valor de entrada
            derivada (bool, optional): Si es True, calcula la derivada. Defaults to False.
        Retorna: 
            np.array: valor de salida
        """
        x = np.clip(x, -50, 50)
        
        sig = 1 / (1 + np.exp(-x))
        
        if derivada:
            return sig * (1 - sig)
        return sig


    def propagacion_adelante(self, entrada):
        """
        Propagación hacia adelante a implementar
        Args:
            entrada (np.array): vector de entrada
        Retorna: 
            salidas (tuple): (salida_capas, activaciones_capas)
        """
        # Convertir la entrada en un array 2D columna si es necesario
        entrada = np.array(entrada).reshape(-1, 1)
        
        # Almacenar todas las salidas y activaciones de cada capa
        salidas = [entrada] # La capa de entrada es la misma
        activaciones = [entrada] # Para cada capa de entrada, activación = entrada
        
        
        for i in range(self.num_capas - 1):
            # Calculas la entrada neta para la siguiente capa
            # z = W * a + b
            """
            W: pesos de la capa actual
            a: activación de la capa anterior
            b: sesgo de la capa actual
            z: entrada neta a la capa actual
            """
            W = self.pesos[i]
            a = activaciones[-1]
            b = self.sesgos[i]
            
            z = np.dot(W, a) + b
            salidas.append(z)
            
            if i== self.num_capas - 2:
                a = z # Es decir la capa de salida es lineal
            else:
                # Aplicar funcion de activacion
                a = self.funcion_activacion(z)
                
            
            activaciones.append(a)

        return salidas, activaciones
            

        
    
    
    
    def propagacion_atras(self,entrada, objetivo, salidas, activaciones):
        """
        Realizar la propagación hacia atrás para calcular los gradientes de los pesos y sesgos
        Args:
            entrada (np.array): vector de entrada
            objetivo (np.array): vector objetivo
            salidas (list): lista de salidas de cada capa
            activaciones (list): lista de activaciones de cada capa
        Retorna:
            tuple: (gradientes_pesos, gradientes_sesgos)
        """
        # Convertir el objetivo en un array 2D
        objetivo = np.array(objetivo, ndmin=2).T
        
        # Inicializar listas para los gradinetes
        gradientes_pesos = [np.zeros_like(peso) for peso in self.pesos]
        gradientes_sesgos = [np.zeros_like(sesgo) for sesgo in self.sesgos]
        
        # Calcular el error en la capa de salida
        #error = (activaciones[-1] - objetivo) * self.funcion_activacion(salidas[-1], derivada=True)
        error = (activaciones[-1] - objetivo) # Capa de salida lineal (no se multiplica por derivada para que no se anule)
        
        # Almacenar gradientes para la última capa
        gradientes_pesos[-1] = np.dot(error, activaciones[-2].T)
        gradientes_sesgos[-1] = error
        
        
        # Propagar el error hacia atrás
        # SECCIÓN PARALELIZABLE: Calculo de errores por capa puede ser paralelizado
        for layer in range(2, self.num_capas):
            z = salidas[-layer]
            derivada_activacion = self.funcion_activacion(z, derivada=True)
            
            error = np.dot(self.pesos[-layer+1].T, error) * derivada_activacion
            
            # Calcular gradientes
            gradientes_pesos[-layer] = np.dot(error, activaciones[-layer-1].T)
            gradientes_sesgos[-layer] = error
            
        return gradientes_pesos, gradientes_sesgos
    
    
    def actualizar_pesos(self, gradientes_pesos, gradientes_sesgos):
        """
        Actualiza los pesos y sesgos usando los gradientes calculados
        Args:
            gradientes_pesos (list): lista de gradientes de pesos
            gradientes_sesgos (list): lista de gradientes de sesgos
        """
        
        # SECCIÓN PARALELIZABLE: Actualización de pesos y sesgos puede ser paralelizado
        for i in range(len(self.pesos)):
            self.pesos[i] -= self.tasa_aprendizaje * gradientes_pesos[i]
            self.sesgos[i] -= self.tasa_aprendizaje * gradientes_sesgos[i]
    
    
    def entrenar(self, entradas_entrenamiento, objetivos_entrenamiento, epocas, mostrar_progreso = True):
        """
        Entrenar la red neuronal usando backpropagation
        Args:
            entradas_entrenamiento (list): lista de vectores de entrada
            objetivos_entrenamiento (list): lista de vectores objetivo
            epocas (int): número de épocas para entrenar
            mostrar_progreso (bool, optional): Si es True, muestra el progreso. Defaults to True.
        """
        num_muestras = len(entradas_entrenamiento)
        
        
        for epoca in range(epocas):
            error_total = 0
            inicio_epoca = time.time()
            
            # SECCIÓN PARALELIZABLE: Entrenamiento por épocas puede ser paralelizado
            for i in range(num_muestras):
                salidas, activaciones = self.propagacion_adelante(entradas_entrenamiento[i])
                
                # Calcular el error cuadrático medio
                objetivo = np.array(objetivos_entrenamiento[i], ndmin=2).T
                error_total += np.mean((activaciones[-1] - objetivo) ** 2)
                
                # Propagación hacia atrás
                gradientes_pesos, gradientes_sesgos = self.propagacion_atras(
                    entradas_entrenamiento[i],
                    objetivos_entrenamiento[i],
                    salidas,
                    activaciones
                )
                
                # Actualizar pesos
                self.actualizar_pesos(gradientes_pesos, gradientes_sesgos)
            fin_epoca = time.time()
            
            # Mostrar progreso
            if mostrar_progreso:
                print(f"Época {epoca+1}/{epocas} - Error: {error_total/num_muestras:.6f} - Tiempo: {fin_epoca - inicio_epoca:.2f}s")
    
    
    def predecir(self, entrada):
        """
        Realiza una predicción para una entrada dada
        Args:
            entrada (np.array): vector de entrada
        Retorna:
            np.array: vector de salida predicho
        """
        _, activaciones = self.propagacion_adelante(entrada)
        return activaciones[-1]


    def calcular_precision(self, entradas_prueba, objetivos_prueba, umbral=0.5):
        predicciones = []
        for entrada in entradas_prueba:
            salida = self.predecir(entrada)
            predicciones.append((salida >= umbral).astype(int))
        predicciones = np.array(predicciones).squeeze()
        objetivos = np.array(objetivos_prueba).squeeze()
        precision = np.mean(predicciones == objetivos)
        return precision
    

if __name__ == "__main__":
    def fahrenheit_to_celsius(fahrenheit: float) -> float:
        celsius = (fahrenheit - 32) * (5 / 9)
        return celsius

    def normalizar_datos(x_train, y_train):
        x_train = np.array(x_train, dtype=float)
        y_train = np.array(y_train, dtype=float)

        # Guardamos min y max para poder desnormalizar después
        x_min, x_max = x_train.min(), x_train.max()
        y_min, y_max = y_train.min(), y_train.max()

        # Normalizamos a [-1, 1]
        x_norm = 2 * (x_train - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (y_train - y_min) / (y_max - y_min) - 1

        # Regresamos también los parámetros para desnormalizar
        return (
            x_norm.reshape(-1, 1),
            y_norm.reshape(-1, 1),
            x_min, x_max, y_min, y_max
        )

    def normalizar_x(x, x_min, x_max):
        x = np.array(x, dtype=float)
        return 2 * (x - x_min) / (x_max - x_min) - 1

    def desnormalizar_y(y_norm, y_min, y_max):
        # Inversa de la normalización a [-1, 1]
        return ( (y_norm + 1) / 2 ) * (y_max - y_min) + y_min

    def generador(min_val, max_val, numero_muestras):
        x_train_fahrenheit = np.linspace(min_val, max_val, numero_muestras)
        y_celsius = fahrenheit_to_celsius(x_train_fahrenheit)
        return x_train_fahrenheit, y_celsius
        
    # ========= PARAMETROS =========
    rango = 200
    numero_muestras = 800
    epocas = 1000
    lr = 0.01   # más pequeña, más estable
    
    # Datos en escala original (F y C)
    x_raw, y_raw = generador(min_val=-rango, max_val=rango, numero_muestras=numero_muestras)

    # Normalizamos para entrenar
    x_train, y_labels, x_min, x_max, y_min, y_max = normalizar_datos(x_raw, y_raw)

    # Arquitectura: 1 entrada, 1 salida, con 1 capa oculta pequeña
    topologia = [1,5,1]  
    red = RedNeuronalBackpropagation(topologia, tasa_aprendizaje=lr)

    # Entrenamiento
    red.entrenar(x_train, y_labels, epocas=epocas, mostrar_progreso=True)

    # Prueba con un valor específico (4°F)
    f_test = 4.0
    x_prueba_norm = normalizar_x([[f_test]], x_min, x_max)
    pred_norm = red.predecir(x_prueba_norm)[0][0]
    pred_celsius = desnormalizar_y(pred_norm, y_min, y_max)
    real_celsius = fahrenheit_to_celsius(f_test)
    error = abs(pred_celsius - real_celsius)
    print(f"\nPredicción para Fahrenheit={f_test} → Celsius_pred={pred_celsius:.2f}, Celsius_real={real_celsius:.2f}, error={error:.2f}")

    # Mostrar varios resultados
    fahrenheit_list = [0, 32, 50, 75, 100, 120, 150, 180, 200]
    print("\nResultados:")
    for f in fahrenheit_list:
        x_f_norm = normalizar_x([[f]], x_min, x_max)
        pred_norm = red.predecir(x_f_norm)[0][0]
        pred_c = desnormalizar_y(pred_norm, y_min, y_max)
        y_real = fahrenheit_to_celsius(f)
        error = abs(pred_c - y_real)
        print(f"Fahrenheit={f:5.1f}, Celsius_real={y_real:7.3f}, Celsius_pred={pred_c:7.3f}, error={error:7.3f}")
