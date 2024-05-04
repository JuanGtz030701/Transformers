import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta
import warnings
import sys
import os
warnings.filterwarnings('ignore')

ruta_directorio_superior = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ruta_directorio_superior)
from functions import normalize, create_transformer, backtesting_final

data_train = pd.read_csv("./data/aapl_5m_train.csv").dropna()
data_test = pd.read_csv("./data/aapl_5m_test.csv").dropna()

train_mean = data_train.loc[:, ["Open", "High", "Low", "Close"]].mean()
train_std = data_train.loc[:, ["Open", "High", "Low", "Close"]].std()

norm_data_train = (data_train.loc[:, ["Open", "High", "Low", "Close"]] - train_mean) / train_std
norm_data_test = (data_test.loc[:, ["Open", "High", "Low", "Close"]] - train_mean) / train_std

""""
Already compiled

data_train = pd.read_csv("./data/aapl_5m_train.csv").dropna()
data_test = pd.read_csv("./data/aapl_5m_test.csv").dropna()

train_mean = data_train.loc[:, ["Open", "High", "Low", "Close"]].mean()
train_std = data_train.loc[:, ["Open", "High", "Low", "Close"]].std()

norm_data_train = (data_train.loc[:, ["Open", "High", "Low", "Close"]] - train_mean) / train_std
norm_data_test = (data_test.loc[:, ["Open", "High", "Low", "Close"]] - train_mean) / train_std

# RSI configurations
rsi_params = {
    'window': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
}

# WMA (Weighted Moving Average) configurations
wma_params = {
    'window': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
}

# MACD configurations
macd_params = {
    'fast_period': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    'slow_period': [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    'signal_period': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
}

# Bollinger Bands configurations
boll_params = {
    'window': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
    'window_dev': [2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4]
}


for i in range(15):  # Cambia el rango según el número de configuraciones que desees procesar
    # RSI
    data_train[f'rsi_{i}'] = ta.momentum.RSIIndicator(data_train['Close'], window=rsi_params['window'][i]).rsi()
    data_test[f'rsi_{i}'] = ta.momentum.RSIIndicator(data_test['Close'], window=rsi_params['window'][i]).rsi()

    # WMA
    data_train[f'wma_{i}'] = ta.trend.WMAIndicator(data_train['Close'], window=wma_params['window'][i]).wma()
    data_test[f'wma_{i}'] = ta.trend.WMAIndicator(data_test['Close'], window=wma_params['window'][i]).wma()

    # MACD
    macd = ta.trend.MACD(data_train['Close'], window_fast=macd_params['fast_period'][i], window_slow=macd_params['slow_period'][i], window_sign=macd_params['signal_period'][i])
    data_train[f'macd_{i}'] = macd.macd()
    data_test[f'macd_{i}'] = ta.trend.MACD(data_test['Close'], window_fast=macd_params['fast_period'][i], window_slow=macd_params['slow_period'][i], window_sign=macd_params['signal_period'][i]).macd()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data_train['Close'], window=boll_params['window'][i], window_dev=boll_params['window_dev'][i])
    data_train[f'bollinger_mavg_{i}'] = bollinger.bollinger_mavg()
    data_train[f'bollinger_hband_{i}'] = bollinger.bollinger_hband()
    data_train[f'bollinger_lband_{i}'] = bollinger.bollinger_lband()

    bollinger_test = ta.volatility.BollingerBands(data_test['Close'], window=boll_params['window'][i], window_dev=boll_params['window_dev'][i])
    data_test[f'bollinger_mavg_{i}'] = bollinger_test.bollinger_mavg()
    data_test[f'bollinger_hband_{i}'] = bollinger_test.bollinger_hband()
    data_test[f'bollinger_lband_{i}'] = bollinger_test.bollinger_lband()

    # Normalize each new indicator and concatenate
    for col in [f'rsi_{i}', f'wma_{i}', f'macd_{i}', f'bollinger_mavg_{i}', f'bollinger_hband_{i}', f'bollinger_lband_{i}']:
        data_train[col] = normalize(data_train, col)
        data_test[col] = normalize(data_test, col)
        norm_data_train = pd.concat([norm_data_train, data_train[col]], axis=1)
        norm_data_test = pd.concat([norm_data_test, data_test[col]], axis=1)


norm_data_train.dropna(inplace=True)
norm_data_test.dropna(inplace=True)

# Export data_train to a CSV file
norm_data_train.to_csv("data_train.csv", index=False)

# Export data_test to a CSV file
norm_data_test.to_csv("data_test.csv", index=False)

"""

norm_data_train = pd.read_csv("./data/data_train.csv")

norm_data_test = pd.read_csv("./data/data_test.csv")

lags = 5

X_train = pd.DataFrame()
X_test = pd.DataFrame()

for lag in range(lags):
    # Add original features with lags
    X_train[f"Close_{lag}"] = norm_data_train.Close.shift(lag)
    
    X_test[f"Close_{lag}"] = norm_data_test.Close.shift(lag)

for i in range(15):  # Suponiendo que tienes 15 configuraciones de cada indicador
    X_train[f'rsi_{i}'] = norm_data_train[f'rsi_{i}']
    X_train[f'wma_{i}'] = norm_data_train[f'wma_{i}']
    X_train[f'macd_{i}'] = norm_data_train[f'macd_{i}']
    X_train[f'bollinger_mavg_{i}'] = norm_data_train[f'bollinger_mavg_{i}']

    X_test[f'rsi_{i}'] = norm_data_test[f'rsi_{i}']
    X_test[f'wma_{i}'] = norm_data_test[f'wma_{i}']
    X_test[f'macd_{i}'] = norm_data_test[f'macd_{i}']
    X_test[f'bollinger_mavg_{i}'] = norm_data_test[f'bollinger_mavg_{i}']
    

    

# Cálculo de las variables objetivo para el entrenamiento y prueba
Y_train = np.where(X_train['Close_0'].shift(-5) > X_train['Close_0'] * (1.01), 2,
                   np.where(X_train['Close_0'].shift(-5) < X_train['Close_0'] * (0.99), 1, 0))
Y_train = pd.DataFrame(Y_train, index=X_train.index)

Y_test = np.where(X_test['Close_0'].shift(-5) > X_test['Close_0'] * (1.01), 2,
                  np.where(X_test['Close_0'].shift(-5) < X_test['Close_0'] * (0.99), 1, 0))
Y_test = pd.DataFrame(Y_test, index=X_test.index)

# Removing NaNs and the last value due to shifting
#X_train.dropna(inplace=True)
copy_train = X_train.copy()
copy_test = X_test.copy()
#X_test.dropna(inplace=True)
X_train = X_train.iloc[lags:-1, :].values
X_test = X_test.iloc[lags:-1, :].values

Y_train = Y_train.iloc[lags:-1].values.reshape(-1, 1)
Y_test = Y_test.iloc[lags:-1].values.reshape(-1, 1)

# Plot train
plt.plot(copy_train.index, copy_train.Close_0)
plt.xlabel('5 min range')
plt.ylabel('Standardized price')
plt.title('Close through time, train data')
plt.show()

# Plot test
plt.plot(copy_test.index, copy_test.Close_0)
plt.xlabel('5 min range')
plt.ylabel('Standardized price')
plt.title('Close through time, test data')
plt.show()


Y_train_valid = pd.DataFrame(Y_train)
Y_test_valid = pd.DataFrame(Y_test)
print(Y_train_valid.value_counts())
print(Y_test_valid.value_counts())

## RESHAPING TENSORS
features = X_train.shape[1]

X_train = X_train.reshape(-1, features, 1)
X_test = X_test.reshape(-1, features, 1)


#Already compiled. TRANSFORMER
## CLASSIFICATION MODEL

input_shape = X_train.shape[1:]

# Hyperparams
head_size = 256
num_heads = 4
num_transformer_blocks = 4
dnn_dim = 4
units = 128


# Defining input_shape as Input layer
input_layer = tf.keras.layers.Input(input_shape)

# Creating our transformers based on the input layer
transformer_layers = input_layer

for _ in range(num_transformer_blocks):
    # Stacking transformers
    transformer_layers = create_transformer(inputs=transformer_layers,
                                            head_size=head_size,
                                            num_heads=num_heads,
                                            dnn_dim=dnn_dim)

# Adding global pooling
pooling_layer = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")\
                                                      (transformer_layers)

# Adding MLP layers
l1 = tf.keras.layers.Dense(units=128, activation="leaky_relu")(pooling_layer)
l2 = tf.keras.layers.Dropout(0.3)(l1)
l3 = tf.keras.layers.Dense(units=128, activation="leaky_relu")(l2)

# Last layer, units = 3 for True and False values
outputs = tf.keras.layers.Dense(units=3, activation="softmax")(l3)

# Model
model = tf.keras.Model(inputs=input_layer,
                       outputs=outputs,
                       name="transformers_classification")

metric = tf.keras.metrics.SparseCategoricalAccuracy()
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss",
#                                              patience=10,
#                                              restore_best_weights=True)]

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=adam_optimizer,
    metrics=[metric],
)

model.summary()


model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    #callbacks=callbacks,
)

model.save("transformer_classifier.keras")


model = tf.keras.models.load_model("transformer_classifier.keras")

# Train results
y_hat_train = model.predict(X_train)
print('Train 0s: ', sum(y_hat_train.argmax(axis=1) == 0))
print('Train 1s: ',sum(y_hat_train.argmax(axis=1) == 1))
print('Train 2s: ',sum(y_hat_train.argmax(axis=1) == 2))

# Test results
y_hat_test = model.predict(X_test)
print('Test 0s: ', sum(y_hat_test.argmax(axis=1) == 0))
print('Test 1s: ',sum(y_hat_test.argmax(axis=1) == 1))
print('Test 2s: ',sum(y_hat_test.argmax(axis=1) == 2))


close_0_standardized = copy_test['Close_0']

# Utilizando la misma media y desviación estándar que para la columna "Close" de X_test
mean_close = train_mean["Close"]
std_close = train_std["Close"]

# Revertir la estandarización
close_0_original = close_0_standardized * std_close + mean_close

copy_test['Close_0'] = close_0_original

data_backtesting = copy_test

predicted_classes = y_hat_test.argmax(axis=1)

predicted_classes_series = pd.Series(predicted_classes).reset_index(drop=True)

data_backtesting['Predicted_Class'] = predicted_classes_series

data_backtesting.dropna(inplace=True)

final_portfolio_value, portfolio_value = backtesting_final(copy_test)

combined_df = pd.DataFrame({
        'Close': data_backtesting['Close_0'],  
        'PortfolioValue': portfolio_value
})

# Ruta del archivo CSV donde quieres guardar los datos
#file_path = r"C:/Users/valer/Documents/Equipo Trading/Transformers/data/values_portfolio.csv"
file_path = "data/values_portfolio.csv"

# Guardar el DataFrame en un archivo CSV
combined_df.to_csv(file_path, index=False)

print('combined_df exported')