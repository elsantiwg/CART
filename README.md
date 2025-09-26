# Clasificador de Correos Spam con Árboles de Decisión (CART)

 implementa un clasificador de correos electrónicos utilizando el algoritmo CART (Classification and Regression Trees) mediante la librería `scikit-learn`.  

El objetivo fue entrenar y evaluar el modelo en múltiples configuraciones para medir su desempeño en términos de Exactitud (Accuracy), F1 Score y Z-score

---

##  Procedimiento

1. **Preparación de datos**
   - Se cargó el dataset `emails_spam_dataset_1000.csv`.
   - Se identificó la columna objetivo: `is_spam`.
   - Se imputaron valores faltantes (mediana para numéricos, moda para categóricos).
   - Se codificaron las variables categóricas usando `OrdinalEncoder`.

2. **Modelo**
   - Se utilizó un árbol de decisión (DecisionTreeClassifier) con criterio de impureza Gini.
   - Se variaron los hiperparámetros en cada corrida:
     - `test_size` (0.20 – 0.35)  
     - `max_depth` (None, 3, 5, 7, 10)  
     - `min_samples_split` (2, 5, 10, 20)  
     - `random_state` diferente en cada ejecución

3. **Entrenamiento y evaluación**
   - Se realizaron 60 corridas del experimento.
   - En cada corrida se calcularon:
     - **Accuracy**
     - **F1 Score**
     - **Z-score** de la exactitud (para medir desviaciones respecto al promedio).

4. **Visualización**
   - Se graficaron los resultados de las 60 corridas mostrando Accuracy, F1 y Z-score.

<img width="1249" height="832" alt="image" src="https://github.com/user-attachments/assets/b8df1097-dd2f-4b7d-af50-b7b74cb50cea" />

---

##  Resultados obtenidos

- **Accuracy promedio**: ~ **93.0%**  
- **Desviación estándar Accuracy**: ~ **1.5%**  
- **F1 Score promedio**: ~ **92.9%**  
- **Desviación estándar F1**: ~ **1.5%**  
- **Número de corridas**: 60
- <img width="422" height="87" alt="image" src="https://github.com/user-attachments/assets/13dc031f-02a1-4bd0-98b1-5b6656300186" />


La siguiente figura muestra la evolución de las métricas a lo largo de las corridas:

![Métricas en múltiples corridas](metrics_plots.png)

---

##  Valoración de los resultados

- El modelo es consistente y estable**, alcanzando un rendimiento alto en todas las corridas, con variaciones mínimas.  
- **Accuracy y F1 Score evolucionan de manera paralela**, lo cual indica que las clases (spam y no spam) están razonablemente balanceadas.  
- El **Z-score** muestra que casi todas las corridas están dentro de ±2 desviaciones estándar, lo esperado en un sistema robusto.  
- **Hiperparámetros**:
  - `max_depth=None` puede llevar a sobreajuste en algunos casos, aunque el impacto fue bajo en este dataset.  
  - Valores altos en `min_samples_split` hacen los árboles más conservadores, pudiendo perder algo de precisión.  
  - Un `test_size` mayor reduce los datos de entrenamiento y genera mayor variabilidad en el desempeño.  

---

##  Conclusiones

1. El modelo CART alcanzó un desempeño sobresaliente (~93% de accuracy y F1), demostrando ser una técnica efectiva para clasificación de spam.  
2. Las variaciones en los resultados se explican principalmente por los cambios en los hiperparámetros y en la proporción de datos de entrenamiento vs. prueba.  
3. La baja dispersión (desviación estándar reducida) confirma la robustez y estabilidad del modelo en este dataset.  



