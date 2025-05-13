#Importamos la libreria de numpy
import numpy as np

#Creamos dos variables
#Variable de entrada(Numero de equipos afectados)
X = 2 * np.random.rand(100, 1)
print(X)
print(f'la longitud del conjunto de datos de incidentes {len(X)}')

#Variable de salida(Coste de los incidentes)
Y = 4 * 3 * X +np.random.rand(100, 1)
print(Y)
print(f'la longitud del conjunto de datos costes {len(Y)}')

'''
Visualizacion del conjunto de datos
necesario importacion de matplotlib.pyplot
'''
import matplotlib.pyplot as plt
## %matplotlib inline

plt.figure(figsize=(10, 8))  # Aumenta el ancho a 10 pulgadas y el alto a 8 pulgadas
plt.plot(X, Y, "b.")
plt.xlabel("Número de equipos afectados (u/1000)")
plt.ylabel("Coste de los incidentes (u/10000)")
plt.title("Relación entre equipos afectados y coste de incidentes")
plt.grid(True)
plt.show()
