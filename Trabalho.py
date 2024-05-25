"""
Este script aplica regressão linear para prever os custos de seguro médico (charges)
com base em vários fatores (X) usando o conjunto de dados 'medical_insurance.csv'.
As variáveis categóricas são convertidas em variáveis dummy para serem usadas no modelo.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns

#-----------------------------------------------------------------------------------------------------------------------
# 1 - prepare data

# Carregar o CSV
df = pd.read_csv('medical_insurance.csv')

# Convertendo variáveis categóricas em variáveis dummy (também conhecidas como variáveis indicadoras)
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

# Dividir os dados em variáveis independentes (X) e variável dependente (y) 
# X contém todos os dados, exceto a coluna charges, enquanto y contém apenas a coluna charges.
X = df.drop(columns=['charges'])
y = df['charges']

# Normalizar as variáveis independentes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em conjunto de treinamento e teste
# test_size=0.2 significa que 20% dos dados são reservados para testes
# random_state=42 garante que essa divisão seja reprodutível
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#-----------------------------------------------------------------------------------------------------------------------
# 2 - create and train the model

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo com nossos dados
model.fit(X_train, y_train)

#-----------------------------------------------------------------------------------------------------------------------
# 3 - validate the model using cross-validation

# Avaliar o modelo usando validação cruzada
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("Cross-validation R^2 scores:", cv_scores)
print("Average cross-validation R^2 score:", np.mean(cv_scores))

#-----------------------------------------------------------------------------------------------------------------------
# 4 - check the results

# Obter parâmetros (coeficientes e intercepto)
slope = model.coef_
intercept = model.intercept_

# Imprimir os coeficientes e o intercepto do modelo
print(f"Model Coefficients: {slope}, Intercept: {intercept}")

# Avaliar o modelo
# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular o erro quadrático médio (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calcular o erro absoluto médio (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calcular o coeficiente de determinação (R^2)
r2 = r2_score(y_test, y_pred)

# intercept (quando negativo isso significa que o valor previsto de Y quando X é zero é negativo. Neste caso, a linha de regressão cruza o eixo y abaixo do valor zero.)
intercept = model.intercept_

# Imprimir MSE e R^2 Score
print("Erro quadrático médio (MSE):", mse)
print("Erro absoluto médio (MAE):", mae)
print("R^2 Score:", r2)
print("Intercept:", intercept)

# Análise de Resíduos
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title("Distribuição dos Resíduos (y_test - y_pred)")
plt.xlabel("Resíduos")
plt.show()

plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Linear Regression)")
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# 5 - use the model to predict

# Fazer previsões para todos os dados
y_pred_all = model.predict(X_scaled)

#-----------------------------------------------------------------------------------------------------------------------
# 6 - plot the prediction (red) AND the real data (blue)

# Para a visualização, vamos plotar um gráfico de dispersão dos valores reais vs preditos

# Gráfico de dispersão dos valores preditos (dados de teste)
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted')  # Valores preditos

# Gráfico de dispersão dos valores reais (dados de teste)
plt.scatter(y_test, y_test, alpha=0.6, color='red', marker='x', label='Actual')  # Valores reais

# Linha de previsões perfeitas (onde valor predito = valor real)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Predictions')

# Rotulação dos eixos
plt.xlabel("Actual Values (charges)")
plt.ylabel("Predicted Values (charges)")

# Título do gráfico
plt.title("Actual vs Predicted Values")

# Adicionar legenda
plt.legend()

# Mostrar o gráfico
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# 7 - save the model

# Salvar o modelo treinado para uso futuro
joblib.dump(model, 'Trabalho\linear_regression_model.pkl')
joblib.dump(scaler, 'Trabalho\scaler.pkl')