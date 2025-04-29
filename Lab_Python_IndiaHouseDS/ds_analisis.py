import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from pandas.plotting import scatter_matrix

warnings.filterwarnings('ignore')

# Configuración para mejorar la visualización
plt.style.use('dark_background')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Cargar el dataset
# Asumimos que el dataset está en un CSV llamado 'india_property.csv'
# Esto debe ser ajustado según dónde esté realmente el dataset
try:
    # Intenta cargar desde un archivo (ajusta la ruta si es necesario)
    data = pd.read_csv('/content/house_prices.csv')
except FileNotFoundError:
    # Si el archivo no existe, crear un dataframe de muestra con la estructura descrita
    print("No se encontró el archivo. Creando un dataset de prueba basado en la estructura descrita.")
    data = pd.DataFrame({
        'Index': range(1, 1001),
        'Title': ['Property ' + str(i) for i in range(1, 1001)],
        'Description': ['Description ' + str(i) for i in range(1, 1001)],
        'Amount(in rupees)': np.random.randint(500000, 10000000, 1000).astype(str),
        'Price (in rupees)': np.random.randint(500000, 10000000, 1000),
        'location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad'], 1000),
        'Carpet Area': [str(np.random.randint(500, 3000)) + ' sqft' for _ in range(1000)],
        'Status': np.random.choice(['Ready to move', 'Under Construction'], 1000),
        'Floor': [str(np.random.randint(1, 20)) + ' out of ' + str(np.random.randint(20, 30)) for _ in range(1000)],
        'Transaction': np.random.choice(['New Property', 'Resale'], 1000),
        'Furnishing': np.random.choice(['Furnished', 'Semi-Furnished', 'Unfurnished'], 1000),
        'facing': np.random.choice(['East', 'West', 'North', 'South'], 1000),
        'overlooking': np.random.choice(['Garden', 'Park', 'Main Road'], 1000),
        'Society': [np.random.choice(['Yes', 'No', np.nan], p=[0.4, 0.4, 0.2]) for _ in range(1000)],
        'Bathroom': np.random.randint(1, 5, 1000).astype(str),
        'Balcony': np.random.randint(1, 3, 1000).astype(str),
        'Car Parking': np.random.choice(['1', '2', '3', np.nan], 1000),
        'Ownership': np.random.choice(['Freehold', 'Leasehold', np.nan], 1000),
        'Super Area': [str(np.random.randint(600, 3500)) + ' sqft' for _ in range(1000)],
    })
    # Las columnas 'Dimensions' y 'Plot Area' serán todas NaN en esta muestra

# Exploración inicial del dataset
print("Dimensiones del dataset:", data.shape)
print("\nPrimeras 5 filas del dataset:")
print(data.head())
print("\nInformación del dataset:")
print(data.info())
print("\nEstadísticas descriptivas:")
print(data.describe())

# Preprocesamiento de datos
# Limpiar y convertir 'Price (in rupees)' si es necesario
if data['Price (in rupees)'].dtype == 'object':
    # Convertir precios de string a float
    def clean_price(price):
        if pd.isna(price):
            return np.nan
        if isinstance(price, str):
            # Eliminar caracteres no numéricos
            price = re.sub(r'[^\d.]', '', price)
            try:
                return float(price)
            except:
                return np.nan
        return price
    
    data['Price (in rupees)'] = data['Price (in rupees)'].apply(clean_price)

# Limpieza y conversión de 'Carpet Area'
def extract_area_value(area_str):
    if pd.isna(area_str):
        return np.nan
    if isinstance(area_str, str):
        # Extraer solo los números
        match = re.search(r'(\d+)', area_str)
        if match:
            return float(match.group(1))
    return np.nan

data['Carpet_Area_Value'] = data['Carpet Area'].apply(extract_area_value)

# Extraer el número de baños
def extract_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        match = re.search(r'(\d+)', value)
        if match:
            return float(match.group(1))
    return np.nan

data['Bathroom_Count'] = data['Bathroom'].apply(extract_number)
data['Balcony_Count'] = data['Balcony'].apply(extract_number)

# Extraer información del piso
def extract_floor(floor_str):
    if pd.isna(floor_str):
        return np.nan
    if isinstance(floor_str, str):
        match = re.search(r'(\d+)', floor_str)
        if match:
            return float(match.group(1))
    return np.nan

data['Floor_Number'] = data['Floor'].apply(extract_floor)

# Convertir variables categóricas a numéricas
categorical_features = ['location', 'Status', 'Transaction', 'Furnishing', 'facing', 'overlooking', 'Ownership']
for feature in categorical_features:
    if feature in data.columns:
        le = LabelEncoder()
        # Solo aplicamos a valores no nulos
        non_null_indices = data[feature].notna()
        if non_null_indices.sum() > 0:  # Si hay algún valor no nulo
            data.loc[non_null_indices, feature + '_encoded'] = le.fit_transform(data.loc[non_null_indices, feature])
            data[feature + '_encoded'] = data[feature + '_encoded'].astype(float)  # Convertir a float para unificar tipos

# Verificar valores nulos en las columnas procesadas
print("\nValores nulos en columnas procesadas:")
processed_columns = ['Price (in rupees)', 'Carpet_Area_Value', 'Bathroom_Count', 
                    'Balcony_Count', 'Floor_Number'] + [f + '_encoded' for f in categorical_features if f in data.columns]
print(data[processed_columns].isnull().sum())

# Seleccionar filas con datos completos en las columnas importantes
important_columns = ['Price (in rupees)', 'Carpet_Area_Value', 'Bathroom_Count', 'location_encoded']
data_clean = data.dropna(subset=important_columns)
print("\nTamaño del dataset después de eliminar filas con valores nulos en columnas importantes:", data_clean.shape)

# 2. Seleccionar tres características y observar su relación con la variable de salida
# Definimos la variable dependiente
y = data_clean['Price (in rupees)']

# Calcular correlaciones con variables numéricas
numeric_columns = [col for col in data_clean.columns if data_clean[col].dtype in ['int64', 'float64'] and col != 'Price (in rupees)' and col != 'Index']
correlation_matrix = data_clean[numeric_columns + ['Price (in rupees)']].corr()
print("\nMatriz de correlación con la variable objetivo:")
print(correlation_matrix['Price (in rupees)'].sort_values(ascending=False))

# Seleccionamos las tres características más correlacionadas con el precio
top_features = correlation_matrix['Price (in rupees)'].sort_values(ascending=False)[1:4].index.tolist()
print("\nLas tres características más correlacionadas son:", top_features)

# Verificar valores NaN en las características seleccionadas antes de continuar
print("\nValores NaN en las características seleccionadas:")
print(data_clean[top_features].isnull().sum())

# Asegurarse de que no haya valores NaN en las características seleccionadas
data_clean_model = data_clean.dropna(subset=top_features + ['Price (in rupees)'])
print(f"\nTamaño del dataset después de eliminar filas con NaN en características seleccionadas: {data_clean_model.shape}")

# Crear un dataframe con las características seleccionadas y la variable objetivo
selected_data = data_clean_model[top_features + ['Price (in rupees)']]

# Verificación final de valores NaN antes de escalar
print("\nVerificación final de valores NaN:")
print(selected_data.isnull().sum())

# Visualización de la matriz de dispersión
plt.figure(figsize=(12, 10))
scatter_matrix(selected_data, figsize=(12, 10), diagonal='kde')
plt.suptitle('Matriz de dispersión para las variables seleccionadas', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('scatter_matrix.png')
plt.close()

# Visualizar relaciones individuales con la variable objetivo
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, feature in enumerate(top_features):
    sns.regplot(x=feature, y='Price (in rupees)', data=data_clean_model, ax=axes[i])
    axes[i].set_title(f'Relación entre {feature} y Price (in rupees)')
plt.tight_layout()
plt.savefig('feature_relationships.png')
plt.close()

# 3. Ajustar un modelo de regresión lineal
# Preparar los datos para el modelo
X = data_clean_model[top_features]
y = data_clean_model['Price (in rupees)']

# Aplicar transformación logarítmica a la variable objetivo para mejorar el modelo
print("\nAplicando transformación logarítmica a los precios...")
y_log = np.log1p(y)  # log(1+x) para manejar valores cero
print(f"Rango original de precios: {y.min()} - {y.max()}")
print(f"Rango de precios transformados: {y_log.min()} - {y_log.max()}")

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verificar si hay NaN después del escalado
if np.isnan(X_scaled).any():
    print("\n¡ADVERTENCIA! Aún hay valores NaN después del escalado.")
    # Opción: imputar valores NaN con la media
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_scaled = imputer.fit_transform(X_scaled)
    print("Se han imputado los valores NaN con la media.")
else:
    print("\nNo hay valores NaN después del escalado. Procediendo con el entrenamiento.")

# Función para evaluar el modelo con diferentes proporciones de datos
def evaluate_model_splits(X, y, y_log, test_sizes=[0.3, 0.5, 0.6], random_state=42):
    results = {}
    
    for test_size in test_sizes:
        # Dividir los datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        # También dividimos los datos transformados logarítmicamente
        _, _, y_train_log, y_test_log = train_test_split(
            X, y_log, test_size=test_size, random_state=random_state
        )
        
        # Modelo lineal estándar
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Modelo lineal con datos transformados
        model_log = LinearRegression()
        model_log.fit(X_train, y_train_log)
        y_log_pred = model_log.predict(X_test)
        y_pred_from_log = np.expm1(y_log_pred)  # Convertir de vuelta a escala original
        
        # Métricas modelo estándar
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Métricas modelo transformado
        mse_log = mean_squared_error(y_test, y_pred_from_log)
        r2_log = r2_score(y_test, y_pred_from_log)
        
        results[f'test_size_{test_size}'] = {
            'train_size': 1 - test_size,
            'test_size': test_size,
            'mse': mse,
            'r2': r2,
            'mse_log': mse_log,
            'r2_log': r2_log
        }
    
    return results

# 4. Estimar el error del modelo con diferentes proporciones
split_results = evaluate_model_splits(X_scaled, y, y_log)
print("\nResultados con diferentes proporciones de train/test:")
for split, metrics in split_results.items():
    print(f"\n{split}:")
    print(f"  Train size: {metrics['train_size']:.0%}, Test size: {metrics['test_size']:.0%}")
    print(f"  MSE (modelo estándar): {metrics['mse']:.2f}")
    print(f"  R² (modelo estándar): {metrics['r2']:.4f}")
    print(f"  MSE (modelo log-transformado): {metrics['mse_log']:.2f}")
    print(f"  R² (modelo log-transformado): {metrics['r2_log']:.4f}")

# Visualizar resultados de diferentes proporciones
splits = [result['test_size'] for result in split_results.values()]
mse_values = [result['mse'] for result in split_results.values()]
r2_values = [result['r2'] for result in split_results.values()]
mse_log_values = [result['mse_log'] for result in split_results.values()]
r2_log_values = [result['r2_log'] for result in split_results.values()]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.plot(splits, mse_values, 'o-', color='blue', label='MSE (estándar)')
ax1.plot(splits, mse_log_values, 'o-', color='cyan', label='MSE (log-transformado)')
ax1.set_xlabel('Proporción del conjunto de prueba')
ax1.set_ylabel('MSE', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2.plot(splits, r2_values, 'o-', color='red', label='R² (estándar)')
ax2.plot(splits, r2_log_values, 'o-', color='orange', label='R² (log-transformado)')
ax2.set_ylabel('R²', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('MSE y R² para diferentes proporciones de train/test')
plt.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.tight_layout()
plt.savefig('split_comparisons.png')
plt.close()

# 5. Cambiar el método de optimización (usando SGD)
# Utilizamos una proporción fija (70-30) para comparar métodos de optimización
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
_, _, y_train_log, y_test_log = train_test_split(X_scaled, y_log, test_size=0.3, random_state=42)

# Modelo con regresión lineal estándar (Ordinary Least Squares)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# Modelo OLS con transformación logarítmica
lr_log_model = LinearRegression()
lr_log_model.fit(X_train, y_train_log)
lr_log_pred = lr_log_model.predict(X_test)
lr_pred_from_log = np.expm1(lr_log_pred)
lr_log_mse = mean_squared_error(y_test, lr_pred_from_log)
lr_log_r2 = r2_score(y_test, lr_pred_from_log)

# Modelo con Stochastic Gradient Descent mejorado
sgd_model = SGDRegressor(
    max_iter=10000,       # Más iteraciones para mejor convergencia
    tol=1e-5,             # Tolerancia más baja para mejor precisión
    penalty='l2',         # Regularización L2 (Ridge)
    alpha=0.01,           # Parámetro de regularización
    learning_rate='adaptive',  # Tasa de aprendizaje adaptativa
    eta0=0.01,            # Tasa de aprendizaje inicial
    random_state=42
)
sgd_model.fit(X_train, y_train)
sgd_pred = sgd_model.predict(X_test)
sgd_mse = mean_squared_error(y_test, sgd_pred)
sgd_r2 = r2_score(y_test, sgd_pred)

# Modelo SGD con transformación logarítmica
sgd_log_model = SGDRegressor(
    max_iter=10000,
    tol=1e-5,
    penalty='l2',
    alpha=0.01,
    learning_rate='adaptive',
    eta0=0.01,
    random_state=42
)
sgd_log_model.fit(X_train, y_train_log)
sgd_log_pred = sgd_log_model.predict(X_test)
sgd_pred_from_log = np.expm1(sgd_log_pred)
sgd_log_mse = mean_squared_error(y_test, sgd_pred_from_log)
sgd_log_r2 = r2_score(y_test, sgd_pred_from_log)

print("\nComparación de métodos de optimización:")
print(f"Linear Regression (OLS): MSE = {lr_mse:.2f}, R² = {lr_r2:.4f}")
print(f"Linear Regression (OLS) con log: MSE = {lr_log_mse:.2f}, R² = {lr_log_r2:.4f}")
print(f"SGD Regression: MSE = {sgd_mse:.2f}, R² = {sgd_r2:.4f}")
print(f"SGD Regression con log: MSE = {sgd_log_mse:.2f}, R² = {sgd_log_r2:.4f}")

# 6. Utilizar métodos de regularización (Ridge y Lasso)
# Ridge (L2 regularization)
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

# Ridge con transformación logarítmica
ridge_log_model = Ridge(alpha=1.0, random_state=42)
ridge_log_model.fit(X_train, y_train_log)
ridge_log_pred = ridge_log_model.predict(X_test)
ridge_pred_from_log = np.expm1(ridge_log_pred)
ridge_log_mse = mean_squared_error(y_test, ridge_pred_from_log)
ridge_log_r2 = r2_score(y_test, ridge_pred_from_log)

# Lasso (L1 regularization)
lasso_model = Lasso(alpha=0.1, random_state=42)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

# Lasso con transformación logarítmica
lasso_log_model = Lasso(alpha=0.1, random_state=42)
lasso_log_model.fit(X_train, y_train_log)
lasso_log_pred = lasso_log_model.predict(X_test)
lasso_pred_from_log = np.expm1(lasso_log_pred)
lasso_log_mse = mean_squared_error(y_test, lasso_pred_from_log)
lasso_log_r2 = r2_score(y_test, lasso_pred_from_log)

print("\nComparación con métodos de regularización:")
print(f"Linear Regression (sin regularización): MSE = {lr_mse:.2f}, R² = {lr_r2:.4f}")
print(f"Linear Regression (log): MSE = {lr_log_mse:.2f}, R² = {lr_log_r2:.4f}")
print(f"Ridge (L2): MSE = {ridge_mse:.2f}, R² = {ridge_r2:.4f}")
print(f"Ridge (L2) con log: MSE = {ridge_log_mse:.2f}, R² = {ridge_log_r2:.4f}")
print(f"Lasso (L1): MSE = {lasso_mse:.2f}, R² = {lasso_r2:.4f}")
print(f"Lasso (L1) con log: MSE = {lasso_log_mse:.2f}, R² = {lasso_log_r2:.4f}")

# Comparar coeficientes entre modelos
def compare_coefficients(models, model_names, feature_names):
    coefficients = pd.DataFrame()
    
    for model, name in zip(models, model_names):
        coefficients[name] = model.coef_
    
    coefficients.index = feature_names
    return coefficients

models = [lr_model, sgd_model, ridge_model, lasso_model]
log_models = [lr_log_model, sgd_log_model, ridge_log_model, lasso_log_model]
model_names = ['OLS', 'SGD', 'Ridge', 'Lasso']
coef_comparison = compare_coefficients(models, model_names, top_features)
coef_log_comparison = compare_coefficients(log_models, [f"{name}_log" for name in model_names], top_features)

print("\nComparación de coeficientes entre modelos (escala original):")
print(coef_comparison)

print("\nComparación de coeficientes entre modelos (escala logarítmica):")
print(coef_log_comparison)

# Visualizar coeficientes
plt.figure(figsize=(14, 8))
coef_all = pd.concat([coef_comparison, coef_log_comparison], axis=1)
coef_all.plot(kind='bar')
plt.title('Coeficientes de los diferentes modelos')
plt.xlabel('Características')
plt.ylabel('Valor del coeficiente')
plt.grid(axis='y', alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('coefficient_comparison.png')
plt.close()

# 7. Seleccionar el mejor modelo basado en los resultados
# Seleccionamos el modelo con mejor desempeño
models_performance = {
    'OLS': (lr_model, lr_mse, lr_r2),
    'OLS_log': (lr_log_model, lr_log_mse, lr_log_r2),
    'SGD': (sgd_model, sgd_mse, sgd_r2),
    'SGD_log': (sgd_log_model, sgd_log_mse, sgd_log_r2),
    'Ridge': (ridge_model, ridge_mse, ridge_r2),
    'Ridge_log': (ridge_log_model, ridge_log_mse, ridge_log_r2),
    'Lasso': (lasso_model, lasso_mse, lasso_r2),
    'Lasso_log': (lasso_log_model, lasso_log_mse, lasso_log_r2)
}

# Ordenar por R² (de mayor a menor)
sorted_models = sorted(models_performance.items(), key=lambda x: x[1][2], reverse=True)
best_model_name = sorted_models[0][0]
best_model_info = models_performance[best_model_name]
best_model, best_mse, best_r2 = best_model_info

print(f"\n7. Modelo seleccionado: {best_model_name} con split 70-30")
print("Parámetros del modelo:")
print(f"Coeficientes: {best_model.coef_}")
print(f"Intercepto: {best_model.intercept_}")

# Crear un DataFrame para interpretar los coeficientes
is_log_model = "_log" in best_model_name
coef_df = pd.DataFrame({
    'Característica': top_features,
    'Coeficiente': best_model.coef_
}).sort_values(by='Coeficiente', ascending=False)

print("\nInterpretación de coeficientes:")
print(coef_df)
if is_log_model:
    print("\nNota: Como se trata de un modelo logarítmico, los coeficientes representan cambios porcentuales aproximados en el precio.")

# 8. Métricas finales del modelo seleccionado
if is_log_model:
    best_pred_log = best_model.predict(X_test)
    best_pred = np.expm1(best_pred_log)
else:
    best_pred = best_model.predict(X_test)

final_mse = mean_squared_error(y_test, best_pred)
final_r2 = r2_score(y_test, best_pred)
final_rmse = np.sqrt(final_mse)

print("\n8. Métricas finales del modelo:")
print(f"MSE: {final_mse:.2f}")
print(f"RMSE: {final_rmse:.2f}")
print(f"R²: {final_r2:.4f}")

# Visualizar predicciones vs valores reales
plt.figure(figsize=(10, 8))
plt.scatter(y_test, best_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title(f'Predicciones vs Valores reales ({best_model_name})')

# Limitar los ejes para mejor visualización si hay outliers
percentile_95 = np.percentile(y_test, 95)
plt.xlim(0, percentile_95)
plt.ylim(0, percentile_95)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('predictions_vs_actual.png')
plt.close()

# Visualizar residuos
residuals = y_test - best_pred
plt.figure(figsize=(10, 6))
plt.scatter(best_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title(f'Residuos vs Predicciones ({best_model_name})')

# Limitar los ejes para mejor visualización
plt.xlim(0, percentile_95)
residual_95 = np.percentile(np.abs(residuals), 95)
plt.ylim(-residual_95, residual_95)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residuals.png')
plt.close()

# Histograma de residuos
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title(f'Distribución de residuos ({best_model_name})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residuals_histogram.png')
plt.close()

# 9. Conclusiones
# Extraer características para la conclusión
top_3_features = coef_df['Característica'].tolist()
best_split = "70-30"
best_method = best_model_name.replace('_log', ' (con transformación logarítmica)') if is_log_model else best_model_name
r2_percentage = best_r2 * 100

