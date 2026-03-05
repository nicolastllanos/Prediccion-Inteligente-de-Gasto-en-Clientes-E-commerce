# Reporte Técnico: Predicción Inteligente de Gasto en Clientes E-commerce
**Evaluación del módulo 6: Aprendizaje de Máquina Supervisado**

## 1. Situación Inicial y Objetivo

El Departamento de Analítica Comercial solicitó un modelo predictivo capaz de estimar el **monto promedio de compra anual** (*Yearly Amount Spent*) de los clientes del e-commerce. Se utilizaron técnicas de **Regresión Supervisada** aplicadas sobre el dataset *Ecommerce Customers* (Kaggle), que contiene variables de comportamiento web y fidelidad del cliente.

## 2. Definición del problema

Dado que la variable objetivo (*Yearly Amount Spent*) es numérica y continua, el problema se define como de **Regresión** y no de Clasificación. Un modelo clasificador discretizaría el gasto en categorías, perdiendo la precisión del monto exacto que el área comercial necesita para personalizar ofertas.

## 3. Preprocesamiento aplicado (Lección 3)

- **Valores nulos:** imputados con la mediana mediante `SimpleImputer`, presentes en las columnas `Avg. Session Length` y `Time on App`.
- **Outliers:** detectados y eliminados en `Time on Website` usando el método del Rango Intercuartílico (IQR). Se eliminaron 12 registros atípicos.
- **Escalamiento:** todas las variables numéricas fueron estandarizadas con `StandardScaler` (Z-Score) para garantizar la convergencia correcta de los modelos.
- **División:** 80% entrenamiento / 20% prueba con `random_state=42` para reproducibilidad.

## 4. Validación Cruzada (Lección 2)

Se aplicó K-Folds (K=5) sobre el conjunto de entrenamiento con Regresión Lineal obteniendo un **RMSE promedio de 12.99 +/- 1.73**, con baja varianza entre folds. Adicionalmente, se comparó el error de entrenamiento vs prueba para diagnosticar el nivel de ajuste: **RMSE Entrenamiento: 12.85 / RMSE Prueba: 14.19**. La diferencia es mínima, lo que confirma que el modelo no presenta sobreajuste y generaliza bien a datos no vistos.

## 5. Comparativa de Modelos y Métricas (Lecciones 4, 5 y 6)

| Modelo | R2 | MSE | RMSE | MAE |
|---|---|---|---|---|
| Regresión Lineal Simple | **0.9587** | 201.37 | 14.19 | $9.80 |
| Regresión Polinomial (Grado 2) | 0.9579 | 205.25 | 14.33 | $10.02 |
| KNN Regressor (K=5) | 0.8786 | 592.68 | 24.35 | $18.07 |
| Ridge Optimizada (alpha=0.1) | 0.9587 | 201.40 | 14.19 | $9.80 |
| Lasso Optimizada (alpha=0.1) | 0.9587 | 201.49 | 14.19 | $9.82 |
| Gradient Boosting Optimizado | 0.9241 | 370.46 | 19.25 | $13.97 |

La **Regresión Lineal Simple** obtuvo el mejor desempeño general, con un R2 de 0.9587, explicando el 95.87% de la varianza del gasto. La Regularización Ridge no modificó los resultados porque los datos no presentaban multicolinealidad. El Gradient Boosting, si bien es el modelo más potente en teoría, en este caso tuvo un R2 inferior porque la relación entre variables y gasto es esencialmente lineal.

## 6. Optimización e Hiperparámetros (Lección 7 y 8)

- **Ridge (L2):** `GridSearchCV` encontró alpha=0.1 como valor óptimo. No mejoró frente al modelo lineal sin regularización, confirmando que los coeficientes no presentaban inflación.
- **Lasso (L1):** `GridSearchCV` encontró también alpha=0.1 como óptimo. Las métricas fueron prácticamente idénticas a Ridge, indicando que ninguna variable debía ser descartada por el modelo (ningún coeficiente convergió a cero).
- **Gradient Boosting:** `GridSearchCV` detectó `learning_rate=0.1`, `max_depth=3` y `n_estimators=100` como configuración óptima sobre 3 folds.

## 7. Importancia de Variables

El análisis de Feature Importance del Gradient Boosting reveló:

| Variable | Importancia |
|---|---|
| Length of Membership | 66.51% |
| Time on App | 23.56% |
| Avg. Session Length | 9.67% |
| Time on Website | 0.25% |

La antigüedad del cliente y el tiempo de uso de la aplicación móvil son los factores más determinantes del gasto anual.

## 8. Conclusiones

**Modelo final seleccionado:** Regresión Lineal Simple, por su mayor R2 (0.9587) y menor error de predicción, con un error promedio de apenas $9.80 por cliente.

**Hallazgos de negocio:**
- La **fidelidad del cliente** (`Length of Membership`) es el predictor más poderoso del gasto. Retener clientes debe ser la prioridad estratégica.
- Los clientes que usan la **aplicación móvil** gastan significativamente más que los que navegan por la web. La empresa debería invertir en mejorar la experiencia de su app.
- El **tiempo en el sitio web** es prácticamente irrelevante para predecir el gasto (0.25% de importancia), lo que sugiere que la web funciona más como canal informativo que transaccional.
- Con este modelo, el equipo de marketing puede anticipar el gasto esperado de cada cliente y diseñar campañas segmentadas con montos de oferta ajustados al perfil real del usuario.
