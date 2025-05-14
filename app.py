#Importamos la libreria de numpy
import numpy as np

'''
1-Creacion del conjunto de datos aleatorios
Necesario la instalacion de numpy: pip install numpy
'''
#Creamos dos variables
#Variable de entrada(Numero de equipos afectados)
X = 2 * np.random.rand(100, 1)
#print(X)
print(f'la longitud del conjunto de datos de incidentes {len(X)}')

#Variable de salida(Coste de los incidentes)
Y = 4 * 3 * X +np.random.rand(100, 1)
#print(Y)
print(f'la longitud del conjunto de datos costes {len(Y)}')

'''
2-Visualizacion del conjunto de datos
necesario importacion de matplotlib.pyplot : pip install matplotlib, pip install inline
'''
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))  # Aumenta el ancho a 10 pulgadas y el alto a 8 pulgadas
plt.plot(X, Y, "b.")
plt.xlabel("Número de equipos afectados (u/1000)")
plt.ylabel("Coste de los incidentes (u/10000)")
plt.title("Relación entre equipos afectados y coste de incidentes")
plt.grid(True)
plt.show()

'''
3-Modificacion del conjunto de datos
Necesario la importacion de pandas : pip install pandas
'''
import pandas as pd

ruta_exportacion = 'C:/Users/Victo/Desktop/Curso_Machine_learning_udemy/Regresion_Lineal_Coste_Incidente_de_Seguridad/DataFrame/datos_incidentes.csv'
#Convierte los datos a diccionario
data = {'n_equipos_afectados':X.flatten(), 'coste':Y.flatten()}
df=pd.DataFrame(data)
print(f'Datos no escalados {df.head(10)}')

# Hacemos el escalado de cada uno de los elementos de dataframe
# Escalado del numero de equipos afectados
df['n_equipos_afectados'] =df['n_equipos_afectados'] * 1000
df['n_equipos_afectados'] =df['n_equipos_afectados'].astype('int')

# Escalado del coste
df['coste'] =df['coste'] * 10000
df['coste'] =df['coste'].astype('int')



# Verificamos el número de filas
print(f'Número de filas en el DataFrame: {len(df)}')
print(f'Forma del DataFrame (filas, columnas): {df.shape}')

# Exportamos el DataFrame a un archivo CSV
df.to_csv(ruta_exportacion, index=False, encoding='utf-8')
print(f'Datos escalados {df.head(10)}')

'''
4-Construccion del modelo
Necesario la importacion de : pip install scikit-learn
'''
from sklearn.linear_model import LinearRegression

#Construccion del modelo y ajuste de la funcion hipotesis
lin_reg = LinearRegression()
lin_reg.fit(df['n_equipos_afectados'].values.reshape(-1,1), df['coste'].values)

# Imprimimos el intercepto (b o w0) Parametro Theta 0
intercepto = lin_reg.intercept_
print(f"Intercepto (b): {intercepto}")

# Imprimimos los coeficientes (pendientes o w1, w2, ...) Theta 1
# En este caso, solo tenemos una característica
coeficientes = lin_reg.coef_
print(f"Coeficiente (pendiente): {coeficientes}")

#Prediccion para el valor minimo y maximo del conjunto de datos de entrenamiento
X_min_max = np.array([[df["n_equipos_afectados"].min()],[df["n_equipos_afectados"].max()]])
Y_train_pred = lin_reg.predict(X_min_max)

#Representacion grafica de la funcion hipostesis generada
plt.figure(figsize=(10, 8))  # Aumenta el ancho a 10 pulgadas y el alto a 8 pulgadas
plt.plot(X_min_max, Y_train_pred, "g-")
plt.plot(df['n_equipos_afectados'], df['coste'], "b.")
plt.xlabel("Equipos afectados")
plt.ylabel("Coste del incidente")
plt.show()