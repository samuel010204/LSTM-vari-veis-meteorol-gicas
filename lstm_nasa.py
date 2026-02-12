# ============================================================
# IMPORTS
# ============================================================
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


# ============================================================
# 1. CARREGAR DADOS E CRIAR DATA/HORA
# ============================================================
data = pd.read_csv("Dados_climaticos_nasa.csv")

data["Data/Hora"] = pd.to_datetime(
    data[["YEAR", "MO", "DY", "HR"]]
    .rename(columns={"MO": "month", "DY": "day", "HR": "hour"})
)

print(data.head())


# ============================================================
# 2. VARIÁVEIS TEMPORAIS CÍCLICAS (NOVIDADE)
# ============================================================

# Hora do dia (0–23)
data["hour_sin"] = np.sin(2 * np.pi * data["HR"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["HR"] / 24)

# Dia do ano (1–365)
data["DOY"] = data["Data/Hora"].dt.dayofyear
data["doy_sin"] = np.sin(2 * np.pi * data["DOY"] / 365)
data["doy_cos"] = np.cos(2 * np.pi * data["DOY"] / 365)


# ============================================================
# 3. SELEÇÃO DAS FEATURES E DO ALVO
# ============================================================

FEATURES = [
    "ALLSKY_SFC_SW_DWN",  # irradiância passada
    "RH2M",              # umidade relativa
    "T2M",               # temperatura do ar
    "PRECTOTCORR",       # precipitação
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos"
    
]

dataset = data[FEATURES].values


# ============================================================
# 4. SEPARAÇÃO TREINO (95%) / TESTE (5%)
# ============================================================

training_data_len = int(np.ceil(len(dataset) * 0.95))


# ============================================================
# 5. NORMALIZAÇÃO
# ============================================================

scaler = MinMaxScaler() # mais estável p/ variáveis meteorológicas
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len]


# ============================================================
# 6. CRIAÇÃO DAS JANELAS DE 60 PASSOS
# ============================================================

HORIZON = 24 # Definição do horizon de previsão

X_train, y_train = [], []

for i in range(60, len(training_data)-HORIZON):
    X_train.append(training_data[i-60:i, :])  # TODAS as features
    y_train.append(training_data[i + HORIZON, 0])       # irradiância futura

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train.reshape(
    (X_train.shape[0], X_train.shape[1], len(FEATURES))
)


# ============================================================
# 7. MODELO LSTM
# ============================================================

model = keras.models.Sequential([
    keras.layers.LSTM(128, return_sequences=True,
                      input_shape=(X_train.shape[1], len(FEATURES))),
    keras.layers.LSTM(128, return_sequences=False),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mae",
    metrics=[keras.metrics.RootMeanSquaredError()]
)

model.summary()


# ============================================================
# 8. TREINAMENTO
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=1,
    batch_size=16
)


# ============================================================
# 9. PREPARAÇÃO DO TESTE
# ============================================================

test_data = scaled_data[training_data_len - 60:]
X_test = []
y_test = data["ALLSKY_SFC_SW_DWN"].values[training_data_len + HORIZON:]

for i in range(60, len(test_data) - HORIZON):
    X_test.append(test_data[i-60:i, :])

X_test = np.array(X_test)
X_test = X_test.reshape(
    (X_test.shape[0], X_test.shape[1], len(FEATURES))
)


# ============================================================
# 10. PREVISÕES
# ============================================================

predictions = model.predict(X_test)

# inverter escala apenas da irradiância
dummy = np.zeros((predictions.shape[0], len(FEATURES)))
dummy[:, 0] = predictions[:, 0]
predictions = scaler.inverse_transform(dummy)[:, 0]


# ============================================================
# 11. DATAFRAME DE RESULTADOS
# ============================================================

test_dates = data["Data/Hora"].iloc[training_data_len + HORIZON:].reset_index(drop=True)

df_results = pd.DataFrame({
    "DataHora": test_dates[:len(predictions)],
    "Real": y_test[:len(predictions)],
    "Predito": predictions
})


# ============================================================
# 12. PLOT DE UM DIA ESPECÍFICO
# ============================================================

dia_escolhido = datetime(2025, 6, 1).date()

df_dia = df_results[df_results["DataHora"].dt.date == dia_escolhido]

#tabela de predito e real
df_dia = df_dia.copy()

df_dia["Hora"] = df_dia["DataHora"].dt.strftime("%H:%M")
df_dia["Erro (W/m²)"] = df_dia["Predito"] - df_dia["Real"]
df_dia["Erro abs (W/m²)"] = np.abs(df_dia["Erro (W/m²)"])
df_dia["Erro (%)"] = 100 * df_dia["Erro abs (W/m²)"] / df_dia["Real"].replace(0, np.nan)

print("\nIrradiância hora a hora –", dia_escolhido)
print(
    df_dia[[
        "Hora",
        "Real",
        "Predito",
        "Erro (W/m²)",
        "Erro (%)"
    ]].to_string(index=False, justify="center", float_format="%.2f")
)

#plot erro absoluto
plt.figure(figsize=(12,4))

plt.bar(
    df_dia["Hora"],
    df_dia["Erro abs (W/m²)"]
)

plt.xlabel("Hora")
plt.ylabel("|Erro| (W/m²)")
plt.title(f"Erro absoluto horário – {dia_escolhido}")
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#plot estimação x realidade
plt.figure(figsize=(14, 6))
plt.plot(df_dia["DataHora"], df_dia["Real"], label="Real", marker="o")
plt.plot(df_dia["DataHora"], df_dia["Predito"], label="Estimado", marker="s")

plt.title(f"Irradiância – {dia_escolhido}")
plt.xlabel("Hora")
plt.ylabel("W/m²")
plt.legend()
plt.grid(True)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
