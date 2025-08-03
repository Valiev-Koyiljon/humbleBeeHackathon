# utils.py
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_correlation_matrix(df, figsize=(12, 10), annot=False, cmap="coolwarm"):
    """
    Plots a correlation heatmap for the given DataFrame.

    Args:
        df (pd.DataFrame): The dataframe to compute correlations on.
        figsize (tuple): Size of the plot.
        annot (bool): Whether to annotate the heatmap with correlation values.
        cmap (str): Colormap to use.
    """
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap=cmap, fmt=".2f", square=True, linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def save_predictions(index, actual, predicted, filename="predictions.csv"):
    """Save index, actual target and predicted values to CSV."""
    df = pd.DataFrame({
        'index': index,
        'actual': actual,
        'predicted': predicted
    })
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


def plot_actual_vs_predicted(index, actual, predicted):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(index, actual, label='Actual', marker='o')
    plt.plot(index, predicted, label='Predicted', marker='x')
    plt.xlabel('Index / Date')
    plt.ylabel('Total Calls')
    plt.title('Actual vs Predicted Total Calls')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def postprocess_predictions(preds, y_train, clip=True, round_preds=True):
    """
    Post-process predicted values.
    - clip: clip predictions to [min(y_train), max(y_train)]
    - round_preds: round predictions to nearest integer
    
    Returns processed predictions as numpy array.
    """
    preds = np.array(preds)
    if clip:
        min_val = y_train.min()
        max_val = y_train.max()
        preds = np.clip(preds, min_val, max_val)
    if round_preds:
        preds = np.round(preds)
    return preds

