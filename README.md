# IA-LetsGo — Modelo de Propensión de Compra

Proyecto de Machine Learning que predice la **probabilidad de que un cliente compre más de un coche** (propensión de compra) a partir de su perfil sociodemográfico, su comportamiento como cliente y el histórico de interacción con la marca (campañas, garantías, averías, quejas, revisiones, etc.).

Todo el trabajo (extracción de datos, limpieza, análisis exploratorio, entrenamiento de modelos y aplicación del modelo final) está documentado paso a paso en el notebook [analisis.ipynb](analisis.ipynb).

## ¿Para qué sirve?

El objetivo de negocio es priorizar a qué clientes dirigir acciones comerciales (por ejemplo, campañas para la compra de un segundo vehículo). El modelo asigna a cada cliente una puntuación de 0 a 1 ("Propensión de Compra") que se traduce en tres segmentos:

- **Alta probabilidad** (≥ 0.75)
- **Probabilidad media** (0.4 – 0.75)
- **Baja probabilidad** (< 0.4)

Con esa segmentación, el equipo comercial puede enfocar recursos en los clientes con mayor probabilidad de conversión en lugar de actuar sobre toda la base de clientes por igual.

## ¿Cómo funciona?

El flujo de trabajo, recogido en `analisis.ipynb`, sigue estas fases:

1. **Extracción de datos** ([SQLQuery.sql](SQLQuery.sql))
   Los datos de entrenamiento provienen de una base de datos SQL Server (`usecases.DATAEX.IA_PROPENSITY_TRAIN`) y se exportan a CSV en [data/raw/](data/raw/).

2. **Análisis exploratorio (EDA)**
   Estudio de la distribución de variables como edad, coste de venta, género, número de coches, etc., mediante gráficos (histogramas, scatter plots, gráficos de tarta) con `matplotlib`/`seaborn`.

3. **Limpieza de datos**
   - Imputación de nulos (p. ej. `Zona_Renta` con la moda) y eliminación de filas con datos críticos faltantes.
   - Eliminación de duplicados.
   - Eliminación de outliers en variables continuas mediante el método IQR.

4. **Codificación de variables**
   - **Categóricas nominales** (`PRODUCTO`, `TIPO_CARROCERIA`, `COMBUSTIBLE`, `GENERO`, etc.) → `LabelEncoder`.
   - **Categóricas ordinales** (`Potencia`, `Zona_Renta`, `Averia_grave`) → `OrdinalEncoder` con un orden definido manualmente.
   - Agrupación de provincias en comunidades autónomas para reducir cardinalidad.
   - El resultado limpio se guarda en [data/processed/IA_PROPENSITY_TRAIN.csv](data/processed/IA_PROPENSITY_TRAIN.csv).

5. **Entrenamiento y comparación de modelos**
   La variable objetivo es `Mas_1_coche` (si el cliente tiene más de un coche). Se entrenan y comparan varios algoritmos de clasificación, todos con búsqueda de hiperparámetros (grid search manual), validación cruzada y control de overfitting:
   - Random Forest
   - AdaBoost
   - Gradient Boosting
   - **XGBoost** (modelo final seleccionado)

   Para cada modelo se evalúan métricas como accuracy, F1, recall, precision, ROC-AUC, matriz de confusión y curvas ROC/Precision-Recall.

6. **Modelo final**
   El modelo XGBoost ganador se guarda en [models/xgboost.pkl](models/xgboost.pkl), junto con el detalle de las combinaciones de hiperparámetros probadas en [models/results_xgboost.csv](models/results_xgboost.csv).

7. **Aplicación a nuevos clientes**
   El modelo entrenado se carga (`joblib.load`) y se aplica sobre un nuevo conjunto de datos ([data/processed/IA_PROPENSITY_INPUT.csv](data/processed/IA_PROPENSITY_INPUT.csv)) para generar, por cada cliente, su probabilidad de compra (`Propension_Compra`) y su clasificación en alta/media/baja probabilidad, con visualizaciones de la distribución resultante.

## Estructura del proyecto

```text
IA-LetsGo/
├── analisis.ipynb              # Notebook principal: EDA, limpieza, modelado y aplicación
├── SQLQuery.sql                 # Consulta de origen de los datos de entrenamiento
├── requirements.txt              # Dependencias de Python del proyecto
├── data/
│   ├── raw/                      # Datos originales exportados de la base de datos
│   └── processed/                 # Datos limpios/codificados, listos para modelar o predecir
└── models/
    ├── xgboost.pkl                # Modelo XGBoost entrenado (modelo final)
    └── results_xgboost.csv        # Resultados de la búsqueda de hiperparámetros
```

## Cómo ejecutarlo

1. Crear un entorno virtual e instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Abrir [analisis.ipynb](analisis.ipynb) en Jupyter o VS Code y ejecutar las celdas en orden:
   - Las primeras secciones generan `data/processed/IA_PROPENSITY_TRAIN.csv` y entrenan los modelos.
   - La última sección ("Aplicación del modelo a un nuevo conjunto de datos") carga `models/xgboost.pkl` y calcula la propensión de compra sobre `data/processed/IA_PROPENSITY_INPUT.csv`.

## Variables principales del dataset

| Variable | Descripción |
| --- | --- |
| `Mas_1_coche` | Variable objetivo: si el cliente tiene más de un coche |
| `Edad_Cliente`, `GENERO`, `ESTADO_CIVIL`, `OcupaciOn` | Perfil sociodemográfico |
| `PRODUCTO`, `TIPO_CARROCERIA`, `COMBUSTIBLE`, `Potencia`, `TRANS` | Características del vehículo |
| `FORMA_PAGO`, `COSTE_VENTA` | Datos de la compra |
| `Campanna1/2/3` | Participación en campañas comerciales |
| `REV_Garantia`, `Averia_grave`, `QUEJA_CAC`, `Revisiones` | Historial de postventa y atención al cliente |
| `Zona_Renta`, `Comunidad` | Localización y nivel de renta de la zona |
| `km_anno` | Kilometraje anual estimado |
