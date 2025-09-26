import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# ===============================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ===============================
# Ruta al archivo
path = "emails_spam_dataset_1000.csv"  # cambia si tu archivo está en otra ruta
df = pd.read_csv(path)

# Identificar columna objetivo
label_candidates = ['label', 'target', 'spam', 'is_spam', 'class']
label_col = None
for c in label_candidates:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    label_col = df.columns[-1]  # última columna por defecto

print("Usando como etiqueta:", label_col)
y = df[label_col]
X = df.drop(columns=[label_col])

# Separar numéricas y categóricas
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Imputación de valores
if len(numeric_cols) > 0:
    num_imputer = SimpleImputer(strategy='median')
    X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)
else:
    X_numeric = pd.DataFrame(index=X.index)

if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X[cat_cols]), columns=cat_cols)
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat_encoded = pd.DataFrame(enc.fit_transform(X_cat_imputed), columns=cat_cols)
else:
    X_cat_encoded = pd.DataFrame(index=X.index)

# Dataset final preprocesado
X_prepared = pd.concat([X_numeric, X_cat_encoded], axis=1)

# Convertir etiquetas a numéricas si es necesario
if y.dtype == 'object' or y.dtype.name == 'category' or y.dtype == bool:
    y = pd.factorize(y)[0]

# ===============================
# 2. ENTRENAMIENTO Y EVALUACIÓN
# ===============================
n_runs = 60  # repetir mínimo 50 veces
results = []
rng = np.random.RandomState(42)

for run in range(n_runs):
    # Variación de parámetros en cada corrida
    test_size = rng.choice([0.2, 0.25, 0.3, 0.35])
    random_state = int(rng.randint(0, 10000))
    max_depth = rng.choice([None, 3, 5, 7, 10])
    min_samples_split = rng.choice([2, 5, 10, 20])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_prepared, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Modelo Decision Tree (CART)
    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) if len(np.unique(y)) == 2 else f1_score(y_test, y_pred, average='macro')
    
    results.append({
        'run': run + 1,
        'test_size': test_size,
        'random_state': random_state,
        'max_depth': 'None' if max_depth is None else max_depth,
        'min_samples_split': min_samples_split,
        'accuracy': acc,
        'f1': f1
    })

results_df = pd.DataFrame(results)

# ===============================
# 3. CÁLCULO DE Z-SCORE
# ===============================
acc_mean = results_df['accuracy'].mean()
acc_std = results_df['accuracy'].std(ddof=0) if results_df['accuracy'].std(ddof=0) != 0 else 1.0
results_df['accuracy_zscore'] = (results_df['accuracy'] - acc_mean) / acc_std

# ===============================
# 4. GRAFICAR RESULTADOS
# ===============================
plt.figure(figsize=(10, 6))
plt.plot(results_df['run'], results_df['accuracy'], marker='o', label='Accuracy')
plt.plot(results_df['run'], results_df['f1'], marker='s', label='F1 score')
plt.plot(results_df['run'], results_df['accuracy_zscore'], marker='^', label='Accuracy Z-score')
plt.title('Decision Tree (CART) - Métricas en múltiples corridas')
plt.xlabel('Número de corrida')
plt.ylabel('Valor de la métrica')
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# 5. RESUMEN ESTADÍSTICO
# ===============================
print("Media Accuracy:", acc_mean)
print("Desv. estándar Accuracy:", acc_std)
print("Media F1:", results_df['f1'].mean())
print("Desv. estándar F1:", results_df['f1'].std(ddof=0))

# Guardar resultados a CSV
results_df.to_csv("decision_tree_runs_results.csv", index=False)
print("\nResultados guardados en: decision_tree_runs_results.csv")
