PROTOCOLO DE EJECUCIÓN: INDUSTRIALIZACIÓN DE PROYECTO DE CIENCIA DE DATOS
Este documento instruye al agente para la transición de un entorno experimental (.ipynb) a un entorno productivo (Scripts + MLOps), utilizando los datos en /data/01_raw.

FASE 1: RECONSTRUCCIÓN LÓGICA (HITOS 1 AL 8)
El agente debe ejecutar y consolidar la lógica de los archivos en /notebooks para alcanzar el Producto Mínimo Viable (MVP).

1. Ingesta y Calidad (Links 1, 2 y 3)
Origen: Leer datos exclusivamente de /data/01_raw.

Acción: Ejecutar análisis de integridad, estadísticas descriptivas y visualizaciones (EDA).

Validación: Identificar tipos de datos y asegurar que el "dato que viene del código JS" sea procesado correctamente.

2. Preparación y Modelado (Links 4, 5, 6 y 7)
Acción: Ejecutar las celdas de Feature Engineering (Encoding, Scaling).

Entrenamiento: Entrenar el Baseline y realizar la Model Selection según los experimentos del notebook.

Interpretación: Extraer la importancia de las variables (Feature Importance).

Salida: Serializar el mejor modelo (.pkl) y sus transformadores asociados.

3. Validación de Demo (Link 8)
Acción: Levantar la interfaz de Streamlit.

Hito de Control: La demo debe ser capaz de cargar el modelo y realizar predicciones en tiempo real sobre datos de entrada manuales o de test.

FASE 2: DECONSTRUCCIÓN EN SCRIPTS PRODUCTIVOS (MLOps)
Una vez validado el Streamlit, el agente debe descomponer la lógica de los notebooks en tres scripts de Python independientes en la carpeta /src.

SCRIPT 1: feature_pipeline.py (Procesamiento y Validación)
Objetivo: Automatizar la transformación de datos.

Tareas:

Carga desde /data/01_raw.

Aplicar limpieza y Feature Engineering.

Punto MLOps: Implementar Validación de Datos (Data Validation) para asegurar que el input cumple con el esquema esperado antes de procesar.

Salida: Guardar dataset procesado en /data/02_processed.

SCRIPT 2: train_pipeline.py (Entrenamiento y Seguimiento)
Objetivo: Estandarizar el entrenamiento del modelo.

Tareas:

Carga de datos desde /data/02_processed.

Entrenamiento del modelo final.

Punto MLOps: Implementar Experiment Tracking (MLflow) para registrar métricas, hiperparámetros y la versión del modelo.

Punto MLOps: Ejecutar Model Validation (pruebas de estrés y métricas de error).

Salida: Modelo registrado y listo para servir.

SCRIPT 3: inference_pipeline.py (Servicio y Despliegue)
Objetivo: Exponer el modelo vía API.

Tareas:

Cargar el modelo y transformadores.

Punto MLOps: Desarrollar una API con FastAPI que reciba peticiones POST.

Reconexión: Modificar el Streamlit para que no cargue el modelo localmente, sino que consuma este servicio FastAPI.