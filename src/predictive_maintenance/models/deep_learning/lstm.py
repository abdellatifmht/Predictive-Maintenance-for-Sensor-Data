from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import standardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.predictive_maintenance.models import BaseModel


class LSTMNetwork(nn.Module):
    """
    LSTM network for predictive maintenance.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the LSTM network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """

        output, _ = self.lstm(x)
        # take the output of the last time step
        last_output = output[:, -1, :]

        return self.fc(last_output).squeeze(1)


class LSTMModel(BaseModel):
    """
    LSTM model for predictive maintenance.

    Input shape: (batch_size, sequence_length, n_)
    """

    def __init__(
            self,
            sequence_length: int = 30,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            learning_rate: float = 0.001,
            epochs: int = 50,
            batch_size: int = 64,
            random_state: int = 42,
            device: str | None = None,
    ):
        self.name = "lstm"

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.device = torch.device(
            device if device is not None
            else (
                "cuda" if torch.cuda.is_available()
                else
                "cpu")
            )

        self.scaler = standardScaler()
        self.model = None
        self.features_columns = None

        torch.manual_seed(random_state)

    def create_sequences(
            self,
            df: pd.DataFrame,
            y: pd.Series,
            features_columns: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sequences of data for LSTM model.

        Args:
            df (pd.DataFrame): Input dataframe.
            y (pd.Series): Target variable.
            features_columns (list[str]): List of feature columns.
        """
        X_sequences = []
        y_sequences = []

        for unit, group in df.groupby("unit"):
            group = group.sort_values("cycle")

            features = group[features_columns].to_numpy(dtype=np.float32)

            targets = y.loc[group.index].to_numpy(dtype=np.float32)

            if len(group) < self.sequence_length:
                continue

            for end in range(self.sequence_length, len(group) + 1):
                start = end - self.sequence_length

                sequence = features[start:end]

                target = targets[end - 1]
                X_sequences.append(sequence)
                y_sequences.append(target)

        if not X_sequences:
            raise ValueError(
                "No sequences were created. "
                "Please check the input data and sequence length."
            )

        return np.array(X_sequences), np.array(y_sequences)

    def _fit_scaler(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """
        Fit the scaler on the training data.

        Args:
            X (np.ndarray): Input data of shape (num_samples, sequence_length, num_features).

        Returns:
            np.ndarray: Scaled input data.
        """
        n_samples, sequence_length, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_2d)
        return X_scaled.reshape(n_samples, sequence_length, n_features)

    def _transform(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """
        Transform the data using the fitted scaler.

        Args:
            X (np.ndarray): Input data of shape (num_samples, sequence_length, num_features).

        Returns:
            np.ndarray: Scaled input data.
        """
        n_samples, sequence_length, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_2d)
        return X_scaled.reshape(n_samples, sequence_length, n_features)

    def _build_model(
            self,
            n_features: int,
    ) -> None:
        """
        Build the LSTM model.

        Args:
            n_features (int): Number of features in the input data.
        """
        self.model = LSTMNetwork(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None,
    ) -> None:
        """
        Train the LSTM model.

        Parameters:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target variable.
            X_val (pd.DataFrame, optional): Validation features. Defaults to None.
            y_val (pd.Series, optional): Validation target variable. Defaults to None.
        """

        self.features_columns = [
            col for col in X.columns if col not in ["unit", "cycle"]
        ]

        if not self.features_columns:
            raise ValueError(
                "No feature columns found. "
                "Please ensure that the input DataFrame contains feature columns."
            )

        X_train_seq, y_train_seq = self.create_sequences(
            X, y, self.features_columns
        )

        X_train_seq = self._fit_scaler(X_train_seq)

        X_val_seq, y_val_seq = None, None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.create_sequences(
                X_val, y_val, self.features_columns
            )
            X_val_seq = self._transform(X_val_seq)

        self._build_model(n_features=len(self.features_columns))

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_train_seq, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()

                predections = self.model(X_batch)

                loss = criterion(predections, y_batch)

                loss.backward()

                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

                train_loss = total_loss / len(dataset)

                if X_val_seq is not None and y_val_seq is not None:
                    val_loss = self.evaluate(X_val_seq, y_val_seq, criterion)
                    print(
                        f"Epoch "
                        f"{epoch + 1:03d}/{self.epochs} "
                        f"- Train Loss: {train_loss:.4f} "
                        f"- Val Loss: {val_loss:.4f}"
                    )

                else:
                    print(
                        f"Epoch "
                        f"{epoch + 1:03d}/{self.epochs} "
                        f"- Train Loss: {train_loss:.4f}"
                    )

        def _evaluate_loss(
                self,
                X: np.ndarray,
                y: np.ndarray,
                criterion: nn.Module = nn.Module,
        ) -> float:
            """
            Calculate validation loss for the model.
            """

            self.model.eval()

            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                predictions = self.model(X_tensor)
                loss = criterion(predictions, y_tensor)

            return loss.item()

        # def predict(
        #         self,
        #         x: pd.DataFrame,
        # ) -> np.ndarray:
        #     """
        #     Predict RUL using the """