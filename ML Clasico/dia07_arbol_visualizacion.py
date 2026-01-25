import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

# =====================
# Dataset (el mismo que ya venís usando)
# =====================
data = {
    "edad":    [22, 25, 23, 30, 21, 24, 28],
    "altura":  [1.70, 1.80, 1.75, 1.90, 1.68, 1.72, 1.78],
    "trabaja": [0, 1, 0, 1, 0, 1, 0],
    "nota":    [7, 8, 6, 9, 10, 5, 4],
}

df = pd.DataFrame(data)

# Variable objetivo
df["aprobado"] = (df["nota"] >= 7).astype(int)

# =====================
# Variables de entrada y salida
# =====================
X = df[["edad", "altura", "trabaja"]]
y = df["aprobado"]

# =====================
# Entrenar árbol con profundidad elegida
# =====================
tree = DecisionTreeClassifier(max_depth=2, random_state=1)
tree.fit(X, y)

# =====================
# Visualizar árbol
# =====================
plt.figure(figsize=(14, 6))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["No aprueba", "Aprueba"],
    filled=True
)
plt.show()
