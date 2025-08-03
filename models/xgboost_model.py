import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

random_seed = 47
np.random.seed(random_seed)

def run_xgboost(train_path, test_path, target_column="total_calls", normalize=False):
    """
    Load data, train XGBoost regression, evaluate, and return the model and predictions.
    
    Args:
        train_path (str): Path to training CSV file.
        test_path (str): Path to testing CSV file.
        target_column (str): Name of the target column.
        normalize (bool): Whether to normalize features with StandardScaler.
        
    Returns:
        model: Trained XGBRegressor model.
        X_test: Test features (possibly scaled).
        y_test: True target values for test.
        y_pred: Predictions on test set.
    """

    # Load data
    train_df = pd.read_csv(train_path, parse_dates=['created_date'])
    test_df = pd.read_csv(test_path, parse_dates=['created_date'])

    # Drop missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Select features (exclude target and date)
    feature_cols = [c for c in train_df.columns if c not in [target_column, 'created_date']]

    X_train = train_df[feature_cols]
    y_train = train_df[target_column]

    X_test = test_df[feature_cols]
    y_test = test_df[target_column]

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train model
    model = XGBRegressor(objective='reg:squarederror', random_state=random_seed, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("📊 Evaluation Metrics:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.4f}%")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

    return model, X_test, y_test, y_pred

