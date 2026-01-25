import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler



# Cargar el conjunto de datos
data = {
    "edad": [22, 25, 23, 30, 21],
    "altura": [1.70, 1.80, 1.75, 1.90, 1.68],
    "trabaja": [0, 1, 0, 1, 0],
    "nota": [7, 8, 6, 9, 10]
}

df = pd.DataFrame(data)
print(df)

#Definir X e Y
# X = variables de entrada (features)
X = df[["edad", "altura", "trabaja"]]

# y = variable a predecir (target)
y = df["nota"]

print("\nX:")
print(X)

print("\ny:")
print(y)

# Normalizar X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Volver a DataFrame para que sea legible
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nX normalizado:")
print(X_scaled)


# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)


print("\nX_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)

# Crear el modelo
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Ver qué aprendió el modelo
print("\nCoeficientes con X normalizado:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

print("Intercepto:", model.intercept_)

# Hacer predicciones
y_pred = model.predict(X_test)

print("\nPredicción del modelo:")
print(y_pred)

print("\nValor real:")
print(y_test.values)

# Calcular el error
mse = mean_squared_error(y_test, y_pred)
print("\nError cuadrático medio (MSE):", mse)
