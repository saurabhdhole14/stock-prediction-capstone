import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_feature_importance(model, feature_names, top_n=10):
    # Requirement: Visualizing top drivers [cite: 84, 93]
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=feature_names)
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(top_n).sort_values().plot(kind='barh')
    plt.title("Top Feature Drivers")
    plt.savefig('artifacts/feature_importance.png')
    plt.close()

def plot_error_histogram(y_true, y_pred):
    # Requirement: Required output artifact [cite: 117]
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50)
    plt.title("Error Histogram")
    plt.savefig('artifacts/error_histogram.png')
    plt.close()

def plot_cumulative_accuracy(y_true, y_pred):
    # Requirement: Cumulative sign-accuracy plot [cite: 117]
    correct = (np.sign(y_true) == np.sign(y_pred)).astype(int)
    cum_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(cum_acc)
    plt.title("Cumulative Sign-Accuracy")
    plt.savefig('artifacts/cumulative_accuracy.png')
    plt.close()