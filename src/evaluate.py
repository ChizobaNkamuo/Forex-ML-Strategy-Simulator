import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

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

def display_metrics(predictions, targets_test):
    print(f"R^2: {r2_score(targets_test, predictions)}")
    directional_accuracy = np.count_nonzero(((predictions > 0) == (targets_test.reshape(-1, 1) > 0)).squeeze())
    print(f"Directional Accuracy: {round(directional_accuracy * 100 / len(targets_test))}%")