import numpy as np

x = np.array([7, 8, 6, 9, 10])

print("x:", x)
print("media:", x.mean())
print("std:", x.std())
print("min/max:", x.min(), x.max())
print("x + 1:", x + 1)
print("x * 2:", x * 2)

z = (x - x.mean()) / x.std()
print("z-score:", z)

# Filtrado de datos (como en datasets reales)
print("Filtrar datos:")
aprobadas = x[x >= 6]
print("aprobadas:", aprobadas)

notas_altas = x[x >= 9]
print("notas altas:", notas_altas)

# Datos como matriz (filas = ejemplos, columnas = features)

print("Matriz datos:")

X = np.array([
    [1.75, 23, 0],  # altura, edad, trabaja (0/1)
    [1.80, 25, 1],
    [1.65, 22, 0],
    [1.90, 30, 1]
])

print("X shape:", X.shape)

altura = X[:, 0]
edad = X[:, 1]
trabaja = X[:, 2]

print("altura:", altura)
print("edad:", edad)
print("trabaja:", trabaja)
print("edad media:", edad.mean())

# Normalización (muy usada en ML)

print("Normalizacion:")

def minmax(col):
    return (col - col.min()) / (col.max() - col.min())

edad_norm = minmax(edad)
altura_norm = minmax(altura)

print("edad_norm:", edad_norm)
print("altura_norm:", altura_norm)

# Día 1 completo: NumPy, filtrado, matrices de datos, normalización

