# Predicción Inteligente de Gasto en Clientes E-commerce
**Evaluación del módulo 6: APRENDIZAJE DE MÁQUINA SUPERVISADO**

Este proyecto implementa un modelo de Machine Learning en Python capaz de predecir el **Gasto Anual** (*Yearly Amount Spent*) de los clientes de una plataforma de e-commerce en función de sus características de navegación y fidelidad.

## Objetivo del Proyecto
El Departamento de Analítica Comercial busca optimizar las estrategias de marketing mediante ofertas hiper-personalizadas. Para lograrlo, este proyecto procesa un dataset de e-commerce, lo limpia, evalúa múltiples algoritmos (desde Regresión Lineal hasta Gradient Boosting) y determina el modelo más robusto para realizar estimaciones precisas del monto de compra esperado por cada cliente nuevo.

---

## Tecnologías Utilizadas
* **Lenguaje:** Python
* **Librerías Principales:**
  * Tratamiento de datos: `pandas`, `numpy`
  * Machine Learning: `scikit-learn`
  * Visualización: `matplotlib`, `seaborn`

---

## Instalación y Uso

1. **Clonar este repositorio:**
```bash
git clone https://github.com/deknar/Prediccion-Inteligente-de-Gasto-en-Clientes-E-commerce.git
cd "proyecto modulo #6"
```

2. **Instalar las dependencias necesarias** (si no las tienes globalmente):
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. **Ejecutar el script principal:**
```bash
python main.py
```
> El script utilizará automáticamente la ruta relativa, cargará los datos, entrenará todos los modelos y lanzará en consola los resultados y las métricas comparativas. Además, generará automáticamente un archivo PNG (`predicciones_vs_realidad.png`) en el mismo directorio.

---

## Pipeline de Machine Learning (Módulo 6)
Este script resuelve progresivamente las lecciones requeridas por el Módulo 6 del Bootcamp de Analítica:

- **Preprocesamiento:** Imputación de valores nulos simulados, detección y limpieza de Outliers (mediante Rango Intercuartil) y estandarización Z-Score (`StandardScaler`).
- **Modelos Base:** Entrenamiento de *Regresión Lineal Simple* y *Polinomial (Grado 2)* con Validación Cruzada (K-Folds = 5).
- **Clasificación vs Regresión:** Demostración empírica comparando `KNeighborsClassifier` frente a `KNeighborsRegressor` para argumentar el uso de variables continuas.
- **Optimización y Ensamble:** Uso de `GridSearchCV` para buscar hiperparámetros óptimos en modelos de Regularización (`Ridge` y `Lasso`) y Árboles Potenciados (`GradientBoostingRegressor`).

---

## Conclusiones del Modelo
Tras la evaluación de múltiples modelos mediante las métricas MAE, RMSE y R2, el algoritmo con mejor desempeño fue la **Regresión Lineal Simple**, con un R2 de **0.9587** (95.87% de la varianza explicada) y un error promedio de apenas **$9.80 por cliente**. La relación entre las variables y el gasto anual resultó ser esencialmente lineal, por lo que añadir complejidad con modelos de ensamble no mejoró los resultados.

El hallazgo de negocio más relevante fue que la **antigüedad del cliente** (`Length of Membership`) explica el 66.51% del gasto, seguida del **tiempo de uso de la app móvil** (23.56%). El tiempo en la web fue prácticamente irrelevante (0.25%).

> Nota: Para ver el desglose completo de métricas y la interpretación de negocio, consultar el archivo [`Informe_Tecnico.md`](./Informe_Tecnico.md) incluido en este repositorio.
