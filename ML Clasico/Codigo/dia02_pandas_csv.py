import pandas as pd

#Parte 2

data = {
    "edad": [22, 25, 23, 30, 21],
    "altura": [1.70, 1.80, 1.75, 1.90, 1.68],
    "trabaja": [0, 1, 0, 1, 0],
    "nota": [7, 8, 6, 9, 10]
}

df = pd.DataFrame(data)
print(df)

#Parte 3

print("\nINFO:")
print(df.info())

print("\nDESCRIBE:")
print(df.describe())

print("\nHEAD:")
print(df.head())

#Parte 4
# Selección de columnas
df_edades = df[["edad", "nota"]]
print("\nSolo edad y nota:")
print(df_edades.head())

# Filtrado de filas
# Filtrar estudiantes aprobados (nota >= 6)
aprobados = df[df["nota"] >= 6]
print("\nAprobados:")
print(aprobados.head())

# Filtrar estudiantes que trabajan (trabaja == 1)
trabajan = df[df["trabaja"] == 1]
print("\nTrabajan:")
print(trabajan.head())

# Filtros combinados
adultos_aprobados = df[(df["edad"] >= 23) & (df["nota"] >= 7)]
print("\nAdultos aprobados:")
print(adultos_aprobados.head())

#Parte 5
# Feature binaria: aprobado (1) / no aprobado (0)
df["aprobado"] = (df["nota"] >= 6).astype(int)

print("\nCon columna 'aprobado':")
print(df.head())

# Feature combinada: adulto y trabaja
df["adulto_trabaja"] = ((df["edad"] >= 23) & (df["trabaja"] == 1)).astype(int)

print("\nCon columna 'adulto_trabaja':")
print(df.head())

# Normalización (z-score)
df["nota_norm"] = (df["nota"] - df["nota"].mean()) / df["nota"].std()

print("\nCon nota normalizada:")
print(df.head())
