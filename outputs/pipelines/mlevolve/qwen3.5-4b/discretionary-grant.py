import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import os
import lightgbm as lgb

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target column
target_col = "AwardTotal"

# Identify feature columns
feature_cols = [col for col in train_df.columns if col != target_col]

# Separate features and target
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].values
X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Create preprocessing pipelines
num_transformer = StandardScaler()
cat_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols),
    ]
)

# Split training data into train and validation FIRST
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Fit preprocessor ONLY on the training split (to prevent leakage)
preprocessor.fit(X_train_split)

# Transform all datasets using the fitted preprocessor
X_train_processed = preprocessor.transform(X_train_split)
X_val_processed = preprocessor.transform(X_val_split)
X_test_processed = preprocessor.transform(X_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_processed)
y_train_tensor = torch.FloatTensor(y_train_split).unsqueeze(1)
X_val_tensor = torch.FloatTensor(X_val_processed)
y_val_tensor = torch.FloatTensor(y_val_split).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_processed)

# Create PyTorch DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Define MLP Model
class RegressionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(RegressionMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.3))

        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.3))

        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        self.layers.append(nn.Softplus())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Initialize model
input_dim = X_train_processed.shape[1]
model = RegressionMLP(input_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# Training loop
model.train()
best_val_loss = float("inf")
patience_counter = 0
max_patience = 15

for epoch in range(100):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    scheduler.step(val_loss.item())

    # Early stopping
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model.pth")
    else:
        patience_counter += 1

    if patience_counter >= max_patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss.item():.6f}"
        )

# Load best model
model.load_state_dict(torch.load("./working/best_model.pth"))
model.eval()

# Predict on validation set
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_predictions = val_outputs.cpu().numpy().flatten()

    # Compute RMSE (official evaluation metric)
    rmse_score = np.sqrt(mean_squared_error(y_val_split, val_predictions))

# Print final validation score
print(f"Final Validation Score: {rmse_score}")

# Predict on test set and save submission
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predictions = test_outputs.cpu().numpy().flatten()

# Create submission file
submission_df = pd.DataFrame(
    {"id": range(len(test_predictions)), "AwardTotal": test_predictions}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print("Submission saved to ./submission/submission.csv")