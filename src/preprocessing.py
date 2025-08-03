# data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from .feature_engineering import add_time_features

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

def run_preprocessing(save_train_test, data_name):
    print("=" * 60)
    print("NYC 311 Complete Data Pipeline")
    print("=" * 60)

    # Load data
    df = pd.read_csv(f"{save_train_test}/{data_name}")
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['date'] = df['created_date'].dt.date

    # Daily total calls
    daily_calls = df.groupby('date').size().reset_index(name='total_calls')
    daily_calls['date'] = pd.to_datetime(daily_calls['date'])

    # Clean boroughs
    df['borough'] = df['borough'].str.upper().str.strip()
    borough_mapping = {
        'MANHATTAN': 'Manhattan', 'BROOKLYN': 'Brooklyn', 'QUEENS': 'Queens',
        'BRONX': 'Bronx', 'STATEN ISLAND': 'Staten Island'
    }
    df['borough'] = df['borough'].map(borough_mapping).fillna('Unknown')

    # Remove duplicates and drop columns with >3.5% nulls
    df = df.drop_duplicates()
    nulls = df.isnull().mean() * 100
    df = df.drop(columns=nulls[nulls > 3.5].index)

    # Drop manually flagged columns
    drop_cols = [
        'location', 'resolution_action_updated_date', 'resolution_description',
        'street_name', 'status', 'agency', 'unique_key']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Merge daily call count
    df['date'] = pd.to_datetime(df['date'])
    merged_df = pd.merge(df, daily_calls, on='date', how='left')
    merged_df = merged_df[['created_date', 'total_calls']].copy()

    # Resample daily
    merged_df = merged_df.set_index('created_date').resample('D').mean()

    # Feature engineering
    daily_df = add_time_features(merged_df)

    # Train-test split
    train_df = daily_df[daily_df.index <= '2025-04-30']
    test_df = daily_df[daily_df.index >= '2025-05-01']

    # Save
    daily_df.to_csv('f"{save_train_test}/preprocessed_all.csv')
    train_df.to_csv(f"{save_train_test}/train.csv")
    test_df.to_csv(f"{save_train_test}/test.csv")

    print("PIPELINE COMPLETE")
    print("Total records:", len(daily_df))
    print("Features:", len(daily_df.columns))
    print("All Data Shape:", daily_df.shape)
    print("Train Data Shape:", train_df.shape)
    print("Test Data Shape:", test_df.shape)
    print("Nulls in training set:")
    print(train_df.isnull().sum())
    print("Nulls in test set:")
    print(test_df.isnull().sum())
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    print("Train shape after removing nulls:", train_df.shape)
    print("Test shape after removing nulls:", test_df.shape)

    return daily_df, train_df, test_df


