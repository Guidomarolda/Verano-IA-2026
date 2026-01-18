import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


# Dataset
data = {
    "edad":    [22, 25, 23, 30, 21, 24, 28],
    "altura":  [1.70,1.80,1.75,1.90,1.68,1.72,1.78],
    "trabaja": [0,   1,   0,   1,   0,   1,   0],
    "nota":    [7,   8,   6,   9,  10,   5,   4]
}

df = pd.DataFrame(data)

# Variable objetivo: aprobado (1) / no aprobado (0)
df["aprobado"] = (df["nota"] >= 7).astype(int)

print(df)

# Variables de entrada
X = df[["edad","altura" ,"trabaja"]]

# Variable objetivo (clasificación)
y = df["aprobado"]

print("\nX:")
print(X)

print("\ny:")
print(y)

# Normalizar X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Volver a DataFrame para claridad
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nX normalizado:")
print(X_scaled)

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y
)


print("\nX_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)

# Crear y entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

print("\nPredicción del modelo:")
print(y_pred)

print("Valor real:")
print(y_test.values)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

# Matriz de confusión
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Métricas extra
print("\nPrecision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

# Arbol de decisión
tree = DecisionTreeClassifier(
    max_depth=2,
    random_state=1
)

tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

print("\nPredicción árbol:")
print(y_pred_tree)

print("Valor real:")
print(y_test.values)

print("Accuracy árbol:", accuracy_score(y_test, y_pred_tree))
print("Confusion Matrix árbol:")
print(confusion_matrix(y_test, y_pred_tree))

# Visualizar árbol
print(
    export_text(
        tree,
        feature_names=list(X_train.columns)
    )
)
