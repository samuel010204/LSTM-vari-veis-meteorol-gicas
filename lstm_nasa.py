# ============================================================
# IMPORTS
# ============================================================
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


# ============================================================
# 1. CARREGAR DADOS E CRIAR DATA/HORA
# ============================================================
data = pd.read_csv("Dados_climaticos_nasa.csv")

# Criar Data/Hora usando YEAR, MO, DY, HR
data["Data/Hora"] = pd.to_datetime(
    data[["YEAR", "MO", "DY", "HR"]]
    .rename(columns={"MO": "month", "DY": "day", "HR": "hour"})
)

# (Opcional) remover colunas antigas
# data = data.drop(columns=["YEAR", "MO", "DY", "HR"])

# Exibir para conferência
print(data.head())
print(data.info())


# ============================================================
# 2. SELEÇÃO DO ALVO
# ============================================================
# Nome da coluna de irradiância do CSV fornecido
TARGET_COLUMN = "ALLSKY_SFC_SW_DWN"

dni = data[[TARGET_COLUMN]]
dataset = dni.values


# ============================================================
# 3. SEPARAÇÃO ENTRE TREINO (95%) E TESTE (5%)
# ============================================================
training_data_len = int(np.ceil(len(dataset) * 0.95))


# ============================================================
# 4. NORMALIZAÇÃO
# ============================================================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len]


# ============================================================
# 5. CRIAR JANELAS DE 60 PASSOS (SLIDING WINDOW)
# ============================================================
X_train, y_train = [], []

for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# reshape para LSTM → (amostras, passos, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))


# ============================================================
# 6. MODELO LSTM
# ============================================================
model = keras.models.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])
model.summary()


# ============================================================
# 7. TREINAMENTO
# ============================================================
history = model.fit(X_train, y_train, epochs=50, batch_size=16)


# ============================================================
# 8. PREPARAÇÃO DO TESTE
# ============================================================
test_data = scaled_data[training_data_len - 60:]
X_test = []
y_test = dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# ============================================================
# 9. PREVISÕES
# ============================================================
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# ============================================================
# 10. CONSTRUIR DF DE RESULTADOS
# ============================================================
test_dates = data["Data/Hora"].iloc[training_data_len:].reset_index(drop=True)

y_test_arr = np.array(y_test).reshape(-1)
pred_arr = predictions.reshape(-1)

df_results = pd.DataFrame({
    "DataHora": test_dates[:len(pred_arr)],
    "Real": y_test_arr[:len(pred_arr)],
    "Predito": pred_arr
})

df_results["DataHora"] = pd.to_datetime(df_results["DataHora"])


# ============================================================
# 11. SELECIONAR DIA PARA PLOTAR
# ============================================================

print("Intervalo TOTAL do dataset:")
print(data["Data/Hora"].min(), "→", data["Data/Hora"].max())

print("\nIntervalo do df_results (previsões):")
print(df_results["DataHora"].min(), "→", df_results["DataHora"].max())


dia_escolhido = datetime(2025, 6, 1).date()   # <-- ALTERE AQUI

df_dia = df_results[df_results["DataHora"].dt.date == dia_escolhido]

if df_dia.empty:
    raise ValueError(f"Nenhum dado encontrado para o dia {dia_escolhido}.")

print(df_dia.head())


# ============================================================
# 12. PLOT DO DIA ESCOLHIDO
# ============================================================
plt.figure(figsize=(14, 6))
day_dates = df_dia["DataHora"]
day_real  = df_dia["Real"]
day_pred  = df_dia["Predito"]

plt.plot(day_dates, day_real, label="Real", marker='o')
plt.plot(day_dates, day_pred, label="Estimativa", marker='s')

plt.title(f"Estimativa vs Real – Irradiância no dia {dia_escolhido}")
plt.ylabel("Irradiância (W/m²)")
plt.xlabel("Horário")
plt.grid(True)
plt.legend()

# Configurar eixo X de 01:00 até 23:00
start_tick = pd.to_datetime(f"{dia_escolhido} 01:00")
end_tick   = pd.to_datetime(f"{dia_escolhido} 23:00")
hour_ticks = pd.date_range(start=start_tick, end=end_tick, freq="1H")

plt.gca().set_xticks(hour_ticks)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.xlim(start_tick, end_tick)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
