import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Fix random seed
torch.manual_seed(47)
np.random.seed(47)

class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(MAPELoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred) / (target + self.eps))) * 100

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# def load_data(train_path, test_path, target_column="total_calls"):
#     train_df = pd.read_csv(train_path, parse_dates=['created_date']).dropna()
#     test_df = pd.read_csv(test_path, parse_dates=['created_date']).dropna()

#     feature_cols = [c for c in train_df.columns if c not in [target_column, 'created_date']]

#     X_train = train_df[feature_cols]
#     y_train = train_df[target_column]

#     X_test = test_df[feature_cols]
#     y_test = test_df[target_column]

#     return X_train, y_train, X_test, y_test

def train_neural_net(train_path, test_path,  target_column="total_calls", normalize = False, epochs=200, batch_size=32, lr=0.001):
    # Load data
    train_df = pd.read_csv(train_path, parse_dates=['created_date'])
    test_df = pd.read_csv(test_path, parse_dates=['created_date'])

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    feature_cols = [c for c in train_df.columns if c not in [target_column, 'created_date']]

    X_train = train_df[feature_cols]
    y_train = train_df[target_column]

    X_test = test_df[feature_cols]
    y_test = test_df[target_column]

    # Normalize if requested
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNN(input_dim=X_train.shape[1]).to(device)
    criterion = MAPELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor.to(device)).cpu().numpy()
        targets = y_test_tensor.numpy()
        mape = mean_absolute_percentage_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        print(f"📊 Evaluation on Test Set:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

    return model, pd.Series(targets.flatten()), preds.flatten()

