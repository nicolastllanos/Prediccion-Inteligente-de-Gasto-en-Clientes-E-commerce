import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score


# =====================================================================
# Proyecto: Predicción Inteligente de Gasto en Clientes E-commerce
# Módulo 6: Aprendizaje de Máquina Supervisado
# =====================================================================
# 
# Situación Inicial: Predecir el monto promedio de compra anual de un
# cliente (Yearly Amount Spent) usando sus características de navegación.
# 
# Objetivo: Diseñar e implementar un modelo predictivo de regresión.


# ---------------------------------------------------------------------
# LECCIÓN 1: Fundamentos del Aprendizaje de Máquina
# ---------------------------------------------------------------------
# Como la variable a predecir ('Yearly Amount Spent') es numérica y 
# continua, estamos frente a un problema de Regresión.

print("Iniciando la carga de datos del e-commerce")
# Usamos os para asegurar que se encuentre el archivo sin importar desde dónde se ejecute el script
# Esto es ideal para la ejecución portable tras clonar desde GitHub
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.join(ruta_actual, 'Ecommerce_Customers.csv')
df = pd.read_csv(ruta_csv)

print("\nPrimeras filas del dataset:")
print(df.head())
print(df.info())


# ---------------------------------------------------------------------
# LECCIÓN 3: Preprocesamiento y escalamiento de datos
# ---------------------------------------------------------------------
# Aquí nos encargamos de tratar valores atípicos, separar los datos y
# aplicar las transformaciones necesarias para que los algoritmos 
# funcionen de manera óptima.

# 1. Tratamiento de valores atípicos (Outliers) en el tiempo de la web
Q1 = df['Time on Website'].quantile(0.25)
Q3 = df['Time on Website'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtramos los datos para quedarnos con datos dentro del límite normal
outliers_mask = (df['Time on Website'] < limite_inferior) | (df['Time on Website'] > limite_superior)
print(f"\nSe han detectado y eliminado {outliers_mask.sum()} outliers en 'Time on Website'.")
df_cl = df[~outliers_mask].copy()

# 2. Selección de las características de estudio (Features)
features_numericos = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
X = df_cl[features_numericos]
y = df_cl['Yearly Amount Spent']

# 3. Separación en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pipeline de transformación: imputamos nulos con la mediana y escalamos
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())                   
])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, features_numericos)]
)

# Aplicamos los cambios
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
print(f"Dimensiones de los datos de entrenamiento procesados: {X_train_scaled.shape}")


# ---------------------------------------------------------------------
# LECCIÓN 2 y 4: Regresiones Lineales y Validación Cruzada
# ---------------------------------------------------------------------
# Evaluaremos la robustez del modelo lineal frente a datos no vistos.

# 1. Modelo Lineal Simple
lin_reg = LinearRegression()

# Validación Cruzada (Dividimos en 5 pliegos para evitar falsas esperanzas)
cv_scores = cross_val_score(lin_reg, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

print(f"\nResultados promedio de Validación Cruzada (RMSE): {cv_rmse.mean():.4f} +/- {cv_rmse.std():.4f}")

# Entrenamos definitivamente y mostramos el peso de las variables
lin_reg.fit(X_train_scaled, y_train)
print("\nImpacto de cada variable frente a la compra (Coeficientes Lineales):")
for i, col in enumerate(features_numericos):
    print(f"  - {col}: {lin_reg.coef_[i]:.4f}")

# Obtenemos predicciones en entrenamiento y prueba para diagnosticar sobreajuste
pred_train_lr = lin_reg.predict(X_train_scaled)
pred_test_lr  = lin_reg.predict(X_test_scaled)

# Nivel de ajuste: si el error de entrenamiento es mucho menor que el de prueba,
# el modelo tiene sobreajuste. Si ambos son altos, hay subajuste.
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train_lr))
rmse_test  = np.sqrt(mean_squared_error(y_test, pred_test_lr))
print(f"\nDiagnostico de ajuste (Regresion Lineal):")
print(f"  RMSE Entrenamiento: {rmse_train:.4f}")
print(f"  RMSE Prueba:        {rmse_test:.4f}")
if rmse_test < rmse_train * 1.2:
    print("  El modelo esta bien ajustado, sin senales claras de sobreajuste.")
else:
    print("  Posible sobreajuste: el error en prueba supera notoriamente al de entrenamiento.")

# 2. Modelo Polinomial (Grado 2)
# Eleva las variables al cuadrado para captar relaciones y curvas orgánicas
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
pred_test_poly = poly_reg.predict(X_test_poly)


# ---------------------------------------------------------------------
# LECCIÓN 5: Algoritmos de Clasificación vs Regresión
# ---------------------------------------------------------------------
# Demostraremos cómo clasificar clientes como "Gasto Alto" vs "Gasto 
# Bajo" no nos sirve frente al poder de predecir el número de su ticket.

# Transformamos temporalmente la compra a categorías alto/bajo
y_train_class = (y_train > y_train.median()).astype(int)
y_test_class = (y_test > y_train.median()).astype(int)

# Entrenamos clasificador de Vecinos Cercanos (KNN)
knn_class = KNeighborsClassifier(n_neighbors=5)
knn_class.fit(X_train_scaled, y_train_class)
pred_class = knn_class.predict(X_test_scaled)
print(f"\nPrecisión aislando High vs Low Ticket (Accuracy): {accuracy_score(y_test_class, pred_class)*100:.2f}%")
print("  => Pero, esta clasificación nos hace perder noción del salto monetario real. Usaremos regresión.")

# Entrenamos el equivalente correcto como Regresor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_train)
pred_test_knn = knn_reg.predict(X_test_scaled)


# ---------------------------------------------------------------------
# LECCIÓN 6: Métricas de Desempeño
# ---------------------------------------------------------------------
# Evaluamos todos los algoritmos creados hasta ahora usando RMSE, MAE y R2.

def evaluar_modelo(y_true, y_pred, nombre_modelo):
    """Función para imprimir de forma limpia las métricas principales."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Resultados del modelo {nombre_modelo}:")
    print(f"  Varianza Explicada (R2): {r2:.4f}")
    print(f"  Error Cuadratico Medio (MSE): {mse:.4f}")
    print(f"  Error Estandarizado (RMSE): {rmse:.4f}")
    print(f"  Error Absoluto (MAE): ${mae:.4f}\n")
    return mae, mse, rmse, r2

print("\nEvaluación y Comparación de Métricas:")
evaluar_modelo(y_test, pred_test_lr, "Regresión Lineal Simple")
evaluar_modelo(y_test, pred_test_poly, "Regresión Polinomial (Grado 2)")
evaluar_modelo(y_test, pred_test_knn, "K-Nearest Neighbors (K=5)")


# ---------------------------------------------------------------------
# LECCIÓN 7 y 8: Optimización, GridSearch y Gradient Boosting
# ---------------------------------------------------------------------
# Mejoraremos nuestros modelos utilizando búsquedas por parámetros y 
# el modelo más avanzado de ensamblado: Boosted Trees.

from sklearn.linear_model import Lasso

# 1a. Regularizacion Ridge
# Penaliza coeficientes grandes (L2), buena para evitar inflacion de pesos.
param_grid_ridge = {'alpha': [0.1, 1.0, 10.0, 100.0]}
grid_ridge = GridSearchCV(Ridge(random_state=42), param_grid_ridge, cv=5, scoring='neg_mean_squared_error')
grid_ridge.fit(X_train_scaled, y_train)

print(f"El mejor hiperparametro detectado para Ridge es: {grid_ridge.best_params_}")
pred_test_ridge = grid_ridge.best_estimator_.predict(X_test_scaled)
evaluar_modelo(y_test, pred_test_ridge, "Regresion Ridge Optimizada")

# 1b. Regularizacion Lasso
# Penaliza coeficientes con L1, puede llevar algunos coeficientes a exactamente cero
# funcionando como selector de features implicito.
param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1.0]}
grid_lasso = GridSearchCV(Lasso(random_state=42, max_iter=5000), param_grid_lasso, cv=5, scoring='neg_mean_squared_error')
grid_lasso.fit(X_train_scaled, y_train)

print(f"El mejor hiperparametro detectado para Lasso es: {grid_lasso.best_params_}")
pred_test_lasso = grid_lasso.best_estimator_.predict(X_test_scaled)
evaluar_modelo(y_test, pred_test_lasso, "Regresion Lasso Optimizada")

# 2. Gradient Boosting (Rúbrica Lección 8)
# Combina muchos árboles de decisión débiles corrigiendo el error de su antecesor
param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}

print("Iniciando la búsqueda de los mejores hiperparámetros para Gradient Boosting...")
grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=3, scoring='r2', n_jobs=-1)
grid_gb.fit(X_train_scaled, y_train)

gb_best = grid_gb.best_estimator_
print(f"Los mejores hiperparámetros detectados para Gradient Boosting son: {grid_gb.best_params_}")
pred_test_gb = gb_best.predict(X_test_scaled)
evaluar_modelo(y_test, pred_test_gb, "Gradient Boosting Optimizado")

# Vemos qué variables fueron decisivas en el algoritmo del árbol
print("\nImportancia de cada variable en la prediccion del Gradient Boosting:")
feature_importances = gb_best.feature_importances_
for i, col in enumerate(features_numericos):
    print(f"  - {col}: {feature_importances[i]*100:.2f}%")


# ---------------------------------------------------------------------
# Tabla comparativa: Valor Real vs Prediccion del Modelo Final
# ---------------------------------------------------------------------
# Aquí se puede ver concretamente a cuánto estimó el modelo el gasto de
# cada cliente del conjunto de prueba, frente a lo que realmente gastó.

tabla_predicciones = pd.DataFrame({
    'Gasto Real ($)':    y_test.values,
    'Prediccion GB ($)': pred_test_gb.round(2),
    'Diferencia ($)':    (y_test.values - pred_test_gb).round(2)
})

print("\nComparacion de predicciones (primeros 15 clientes del conjunto de prueba):")
print(tabla_predicciones.head(15).to_string(index=False))

# ---------------------------------------------------------------------
# EXTRAS: Exportando el Gráfico Final
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Pintamos las predicciones contra los valores reales guardados
sns.scatterplot(x=y_test, y=pred_test_gb, color='#2ecc71', alpha=0.7, edgecolor='k')

# Pintar línea perfecta 1:1 donde Error=0
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)

plt.title("Valores Reales vs Predicciones de Gradient Boosting", fontsize=14, fontweight='bold')
plt.xlabel("Gasto Anual Real ($)", fontsize=12)
plt.ylabel("Gasto Anual Predicho ($)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Guardar y mostrar
ruta_grafico = os.path.join(ruta_actual, "predicciones_vs_realidad.png")
plt.savefig(ruta_grafico, bbox_inches='tight')
print(f"\nSe ha guardado exitosamente el gráfico en la ruta: {ruta_grafico}")
print("El análisis y predicción finalizaron correctamente.\n")
