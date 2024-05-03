import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta
import warnings

warnings.filterwarnings('ignore')

# Normalizing
def normalize(data, column_name):
    return (data[column_name] - data[column_name].mean()) / data[column_name].std()

def create_transformer(inputs, head_size, num_heads, dnn_dim):
    # Stacking layers
    l1 = tf.keras.layers.MultiHeadAttention(key_dim=head_size,
                                            num_heads=num_heads,
                                            dropout=0.2)(inputs, inputs)
    l2 = tf.keras.layers.Dropout(0.2)(l1)
    l3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(l2)
    
    res = l3 + inputs
    
    # Traditional DNN
    l4 = tf.keras.layers.Conv1D(filters=4, kernel_size=1, activation="relu")(res)
    l5 = tf.keras.layers.Dropout(0.2)(l4)
    l6 = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(l5)
    l7 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(l6)
    return l7 + res



def backtesting_final(data):
    # Copia de los datos para evitar modificar el original
    data = data.copy()

    # Parámetros encontrados en proyecto 2 data 5 min:  1.084320935	0.914153619	43	95

    take_profit_multiplier = 1.084320935
    stop_loss_multiplier = 0.914153619
    n_shares_long = 43
    n_shares_short = 95

    # Inicializar variables para el backtesting
    cash = 1_000_000
    margin_account = 0  # Cuenta de margen para operaciones en corto
    active_operations = []
    history = []
    portfolio_value = []
    commission = 0.125 / 100
    margin_call = 0.25  # Porcentaje de margen requerido

    for i, row in data.iterrows():
        # Actualizar el margen necesario para las posiciones cortas
        margin_required = sum(op["n_shares"] * row.Close_0 * margin_call 
                              for op in active_operations if op["type"] == "short")

        if margin_required > margin_account:
            additional_margin_needed = margin_required - margin_account

            if cash >= additional_margin_needed:
                # Si hay suficiente efectivo, transferirlo a la cuenta de margen
                cash -= additional_margin_needed
                margin_account += additional_margin_needed
            else:
                # Si no hay suficiente efectivo, cerrar posiciones cortas hasta que el margen sea suficiente
                for operation in active_operations.copy():
                    if operation["type"] == "short":
                        profit = (operation["sold_at"] - row.Close_0) * operation["n_shares"]
                        cash += profit - (profit * commission)  # Ajustar por comisión
                        margin_account -= row.Close_0 * operation["n_shares"] * margin_call  # Liberar el margen reservado
                        cash+= operation["n_shares"] * row.Close_0 * margin_call
                        history.append({"operation": "closed short", "price": row.Close_0, "n_shares": operation["n_shares"]})
                        active_operations.remove(operation)
                        if sum(op["n_shares"] * row.Close_0 * margin_call for op in active_operations if op["type"] == "short") <= (margin_account+cash):
                            break  # Salir del bucle si el margen es suficiente

                        
        if margin_required < margin_account:
            excess_margin = margin_account - margin_required
            cash += excess_margin
            margin_account -= excess_margin

        # Cerrar operaciones largas y cortas según stop loss y take profit
        for operation in active_operations.copy():
            close_position = False
            if operation["type"] == "long":
                if row.Close_0 <= operation["stop_loss"] or row.Close_0 >= operation["take_profit"]:
                    cash += row.Close_0 * operation["n_shares"] * (1 - commission)
                    close_position = True
            elif operation["type"] == "short":
                if row.Close_0 >= operation["stop_loss"] or row.Close_0 <= operation["take_profit"]:
                    cash += (operation["sold_at"] - row.Close_0) * operation["n_shares"] * (1 - commission)
                    margin_account -= operation["n_shares"] * row.Close_0 * margin_call
                    cash += operation["n_shares"] * row.Close_0 * margin_call
                    close_position = True

            if close_position:
                history.append({"operation": f"closed {operation['type']}", "price": row.Close_0, "n_shares": operation["n_shares"]})
                active_operations.remove(operation)

        # Abrir nuevas operaciones según las señales
        
        # Long
        if cash > row.Close_0 * n_shares_long * (1 + commission):
            if row.Predicted_Class == 2:  
                active_operations.append({
                    "bought_at": row.Close_0,
                    "type": "long",
                    "n_shares": n_shares_long,
                    "stop_loss": row.Close_0 * stop_loss_multiplier,
                    "take_profit": row.Close_0 * take_profit_multiplier
                })
                cash -= row.Close_0 * n_shares_long * (1 + commission)
                history.append({"operation": "opened long", "price": row.Close_0, "n_shares": n_shares_long})

        # Short
        required_margin_for_new_short = row.Close_0 * n_shares_short * margin_call
        if cash >= required_margin_for_new_short:  # Verificar si hay suficiente efectivo para el margen requerido
            if row.Predicted_Class == 1:  # Ejemplo de señal para operación corta
                active_operations.append({
                    "sold_at": row.Close_0,
                    "type": "short",
                    "n_shares": n_shares_short,
                    "stop_loss": row.Close_0 * stop_loss_multiplier,
                    "take_profit": row.Close_0 * take_profit_multiplier,
                })
                margin_account += required_margin_for_new_short
                cash -= required_margin_for_new_short  # Reservar efectivo para el margen
                history.append({"operation": "opened short", "price": row.Close_0, "n_shares": n_shares_short})

        # Actualizar el valor de la cartera
        asset_vals = sum([op["n_shares"] * row.Close_0 for op in active_operations if op["type"] == "long"])
        portfolio_value.append(cash + asset_vals + margin_account)

    final_portfolio_value = portfolio_value[-1]
    return final_portfolio_value, portfolio_value