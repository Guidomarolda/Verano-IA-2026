import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# Reproducible
rng = np.random.default_rng(42)

N = 120

# Features
edad = rng.integers(18, 36, size=N)  # 18 a 35
trabaja = rng.integers(0, 2, size=N)  # 0/1
altura = rng.normal(1.75, 0.08, size=N)  # promedio 1.75m, desv 8cm
altura = np.clip(altura, 1.55, 2.05)

# Nota "realista" (modelo generador)
# base + (edad) - (trabaja) + ruido
ruido = rng.normal(0, 1.2, size=N)

nota = (
    6.5
    + 0.08 * (edad - 24)        # edad aporta un poquito
    - 1.2 * trabaja             # trabajar complica un poco
    + 0.2 * (altura - 1.75)     # altura casi no importa
    + ruido
)

nota = np.clip(nota, 1, 10).round(0).astype(int)

df = pd.DataFrame({
    "edad": edad,
    "altura": altura.round(2),
    "trabaja": trabaja,
    "nota": nota
})

df["aprobado"] = (df["nota"] >= 7).astype(int)

print(df.head())
print("\nDistribucion aprobado:")
print(df["aprobado"].value_counts())

# Variables de entrada y objetivo
X = df[["edad", "altura", "trabaja"]]
y = df["aprobado"]

# CV estratificada y reproducible
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Logistic Regression con normalizacion (Pipeline)
logit = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

logit_scores = cross_val_score(logit, X, y, cv=cv, scoring="accuracy")
print("\nLogisticRegression CV scores:", logit_scores)
print("LogisticRegression CV mean:", logit_scores.mean())

# Entrenar un modelo final (con TODO el dataset) solo para ver coeficientes
logit.fit(X, y)

coefs = logit.named_steps["model"].coef_[0]
intercepto = logit.named_steps["model"].intercept_[0]

print("\nCoeficientes aprendidos (modelo entrenado en TODO el dataset):")
for col, coef in zip(X.columns, coefs):
    print(f"{col}: {coef}")

print("Intercepto:", intercepto)


# Decision Tree (no necesita normalizacion)
tree = DecisionTreeClassifier(max_depth=3, random_state=1)
tree_scores = cross_val_score(tree, X, y, cv=cv, scoring="accuracy")
print("\nDecisionTree CV scores:", tree_scores)
print("DecisionTree CV mean:", tree_scores.mean())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for d in [1, 2, 3, 4, 5, None]:
    tree = DecisionTreeClassifier(max_depth=d, random_state=1)
    scores = cross_val_score(tree, X, y, cv=cv, scoring="accuracy")
    label = f"depth={d}" if d is not None else "depth=None"
    print(f"{label:10s} → scores={scores.round(2)} | mean={scores.mean():.2f}")
