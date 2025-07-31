import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go

RISK_FREE_RATE = 0.02 / 252

def plot_results(predictions, targets_test):
    pred_series = pd.Series(predictions.flatten(), name="Predicted")
    true_series = pd.Series(targets_test, name="Actual")


    plt.figure(figsize=(14, 6))
    plt.plot(true_series, label="Actual")
    plt.plot(pred_series, label="Predicted")
    plt.title("RNN Predictions vs Actual Log Returns")
    plt.xlabel("Time Step")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def directional_accuracy(predictions, targets):
    directional_accuracy = np.count_nonzero((predictions > 0).squeeze() == (targets > 0).squeeze())
    print(f"Directional Accuracy: {round(directional_accuracy * 100 / len(targets))}%")

def display_metrics(predictions, targets_test, data, sequence_length):
    print(f"R^2: {r2_score(targets_test, predictions)}")
    predictions_percent = ((np.exp(predictions) - 1).reshape(-1, 1)).squeeze()
    targets_percent = ((np.exp(targets_test) - 1).reshape(-1, 1)).squeeze()


    split = int(0.8 * len(data))
    data = data[split + sequence_length:]

    take_profit = 0.02
    stop_loss = take_profit / 2
    conditions = [np.abs(predictions_percent) < take_profit, predictions_percent > 0, predictions_percent < 0]
    positions = np.select(conditions, [0, 1, -1]).squeeze()
    
    fee_per_lot = 3  # $3 per lot
    lot_size = 0.01
    contract_value = lot_size * 100000  # Standard forex lot
    fee_percentage = fee_per_lot / (contract_value * data["Open"].to_numpy())
 
    returns = positions * targets_percent
    clipped_returns = np.clip(returns, -stop_loss, take_profit)

    final_returns = clipped_returns - np.abs(positions) * fee_percentage
    
    print(f"Predictions shape: {predictions_percent.shape}")
    print(f"Targets shape: {targets_percent.shape}")
    print(f"Data shape after split: {data.shape}")

    print(f"Mean final returns: {np.mean(final_returns)}")
    print(f"Std final returns: {np.std(final_returns)}")
    print(f"Min/Max returns: {np.min(final_returns)}, {np.max(final_returns)}") 
    epsilon = 1e-8
    excess_returns = final_returns - RISK_FREE_RATE
    sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + epsilon)
    print(f"Sharpe ratio (annualized): {sharpe_ratio * np.sqrt(252)}")


def plot_training_history(dates, predicted):
    plt.plot(dates, predicted[:, 0], label="Pred Open")
    plt.plot(dates, predicted[:, 2], label="Pred High")
    plt.plot(dates, predicted[:, 3], label="Pred Low")
    plt.plot(dates, predicted[:, 1], label="Pred Close")
    plt.legend()
    plt.show()

def plot_candle_sticks(predicted, truth, dates,start = 0, end = 100):
    #["Open", "Close", "High", "Low"]
    print("Sample open values (truth):", truth[:5, 0])
    print("Sample open values (pred):", predicted[:5, 0])
    results = pd.DataFrame({
        "date" : pd.to_datetime(dates),
        "open": truth[:, 0],
        "close": truth[:, 1],
        "high": truth[:, 2],
        "low": truth[:, 3],

        "pred_open": predicted[:, 0],
        "pred_close": predicted[:, 1],
        "pred_high": predicted[:, 2],
        "pred_low": predicted[:, 3],
    })
    results.set_index("date", inplace=True)

    truth_candles = go.Candlestick(
        x=results.index[start:end],
        open=results["open"][start:end],
        high=results["high"][start:end],
        low=results["low"][start:end],
        close=results["close"][start:end],
        name="Actual"
    )

    predicted_candles = go.Candlestick(
        x=results.index[start:end],
        open=results["pred_open"][start:end],
        high=results["pred_high"][start:end],
        low=results["pred_low"][start:end],
        close=results["pred_close"][start:end],
        name="Predicted",
        increasing_line_color="blue",
        decreasing_line_color="orange",
        opacity=0.6
    )



    fig = go.Figure(data=[truth_candles, predicted_candles])
    fig.update_layout(title="Actual vs Predicted Candles", xaxis_title="Date", yaxis_title="Price")
    fig.show()