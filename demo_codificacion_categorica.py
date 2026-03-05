import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =====================================================================
# Demostración: Codificación de Variables Categóricas (Lección 3)
# =====================================================================
#
# Este archivo complementa el proyecto principal (main.py) y demuestra
# el proceso de codificación de variables categóricas, requerido por
# la Lección 3 del Módulo 6.
#
# Dataset utilizado: Ecommerce_Customers.csv
# Variable categórica de ejemplo: Avatar (color de perfil del usuario)

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.join(ruta_actual, 'Ecommerce_Customers.csv')
df = pd.read_csv(ruta_csv)

print("Valores únicos en la columna 'Avatar' (variable categórica):")
print(df['Avatar'].unique()[:10], "... (solo los primeros 10)")
print(f"Total de categorías: {df['Avatar'].nunique()}\n")


# ---------------------------------------------------------------------
# Método 1: Label Encoding
# ---------------------------------------------------------------------
# Convierte cada categoría en un número entero único.
# Útil cuando las categorías tienen un orden implícito (ordinal).
# En este caso lo aplicamos a modo demostrativo.

le = LabelEncoder()
df['Avatar_LabelEncoded'] = le.fit_transform(df['Avatar'])

print("Label Encoding aplicado a 'Avatar':")
print(df[['Avatar', 'Avatar_LabelEncoded']].head(10))
print()


# ---------------------------------------------------------------------
# Método 2: One-Hot Encoding
# ---------------------------------------------------------------------
# Crea una columna binaria por cada categoría.
# Evita imponer un orden artificial entre categorías nominales.
# Es el método preferido para variables sin jerarquía.

avatar_dummies = pd.get_dummies(df['Avatar'], prefix='Avatar')

print(f"One-Hot Encoding generó {avatar_dummies.shape[1]} columnas.")
print("Muestra de las primeras 5 filas y 5 columnas del resultado:")
print(avatar_dummies.iloc[:5, :5])
print()

# Nota: En el modelo principal no se usa 'Avatar' porque tiene 138 categorías
# distintas (alta cardinalidad), lo que generaría un exceso de columnas
# y no aporta valor predictivo comprobado frente al gasto anual del cliente.
print("Nota: La columna 'Avatar' no se incluye en el modelo final por alta cardinalidad.")
