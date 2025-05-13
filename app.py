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

