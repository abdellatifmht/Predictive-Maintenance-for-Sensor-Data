from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMNetwork(nn.Module):
    """Réseau LSTM sous-jacent : quelques couches récurrentes + une petite
    tête feed-forward qui projette le dernier état caché sur un scalaire (RUL).
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


class LSTMModel:
    """Modèle de prédiction de RUL basé sur un LSTM.

    Contrairement aux modèles classiques (LinearRegression, XGBoost, LightGBM),
    ce modèle a besoin de voir plusieurs cycles consécutifs d'un même moteur
    pour faire une prédiction, pas une seule ligne. `X` doit donc TOUJOURS
    contenir les colonnes 'unit' et 'cycle' en plus des features, à la fois
    pour `fit` et pour `predict` — ne pas appeler `prepare_data()` (qui les
    supprime) avant de passer les données à ce modèle.

    Interface volontairement alignée sur les autres modèles du projet
    (`.name`, `.fit`, `.predict`, `.save`, `.load`) pour s'intégrer au reste
    du pipeline avec un minimum de code spécifique.
    """

    name = "LSTM"

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        if sequence_length < 1 or epochs < 1 or batch_size < 1:
            raise ValueError("sequence_length, epochs and batch_size must be positive")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _LSTMNetwork | None = None
        self.feature_columns: list[str] | None = None
        self.feature_mean: pd.Series | None = None
        self.feature_std: pd.Series | None = None

        torch.manual_seed(random_state)

    # ------------------------------------------------------------------
    # Scaling (fit sur le train uniquement, jamais recalculé au predict)
    # ------------------------------------------------------------------

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        return ((X[self.feature_columns] - self.feature_mean) / self.feature_std).to_numpy(
            dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Construction des séquences
    # ------------------------------------------------------------------

    def _build_training_sequences(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Une séquence par cycle de chaque moteur (fenêtre glissante causale,
        rembourrée par répétition de la première ligne si le moteur n'a pas
        encore `sequence_length` cycles d'historique).
        """
        sequences, targets = [], []

        df = X.copy()
        df["_target"] = y.values

        for _, unit_df in df.sort_values("cycle").groupby("unit"):
            values = self._scale(unit_df)
            unit_targets = unit_df["_target"].to_numpy()

            for i in range(len(unit_df)):
                start = max(0, i - self.sequence_length + 1)
                window = values[start : i + 1]

                if len(window) < self.sequence_length:
                    pad = np.repeat(window[:1], self.sequence_length - len(window), axis=0)
                    window = np.concatenate([pad, window], axis=0)

                sequences.append(window)
                targets.append(unit_targets[i])

        return np.stack(sequences), np.array(targets, dtype=np.float32)

    def _build_inference_sequences(self, X: pd.DataFrame) -> tuple[np.ndarray, list]:
        """Une seule séquence par moteur : les `sequence_length` derniers
        cycles disponibles (avec le même remplissage que pour l'entraînement
        si le moteur en a moins).
        """
        sequences, units = [], []

        for unit, unit_df in X.sort_values("cycle").groupby("unit"):
            values = self._scale(unit_df)
            window = values[-self.sequence_length :]

            if len(window) < self.sequence_length:
                pad = np.repeat(window[:1], self.sequence_length - len(window), axis=0)
                window = np.concatenate([pad, window], axis=0)

            sequences.append(window)
            units.append(unit)

        return np.stack(sequences), units

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> LSTMModel:
        if X.empty or len(X) != len(y):
            raise ValueError("Non-empty aligned training features and labels are required")
        if (X_val is None) != (y_val is None):
            raise ValueError("Validation features and labels must be provided together")
        if X_val is not None and (X_val.empty or len(X_val) != len(y_val)):
            raise ValueError("Non-empty aligned validation features and labels are required")
        self.feature_columns = [c for c in X.columns if c not in ("unit", "cycle")]
        self.feature_mean = X[self.feature_columns].mean()
        self.feature_std = X[self.feature_columns].std().replace(0, 1.0).fillna(1.0)

        X_seq, y_seq = self._build_training_sequences(X, y)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model = _LSTMNetwork(
            n_features=len(self.feature_columns),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_seq, y_val_seq = self._build_training_sequences(X_val, y_val)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(loader.dataset)

            if has_val:
                val_loss = self._evaluate_loss(X_val_seq, y_val_seq, criterion)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                    }

                print(
                    f"Epoch {epoch + 1:03d}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1:03d}/{self.epochs} - Train Loss: {train_loss:.4f}")

        # Restaure les poids du meilleur epoch (val loss la plus basse) plutôt
        # que de garder ceux du dernier epoch, potentiellement sur-appris.
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best epoch (val loss: {best_val_loss:.4f}).")

        return self

    def _evaluate_loss(self, X_seq: np.ndarray, y_seq: np.ndarray, criterion: nn.Module) -> float:
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.from_numpy(X_seq).to(self.device)
            y_tensor = torch.from_numpy(y_seq).to(self.device)
            loss = criterion(self.model(X_tensor), y_tensor)

        return loss.item()

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Renvoie une prédiction par moteur présent dans `X`, dans l'ordre
        croissant des identifiants de moteur (comportement de `groupby`).
        """
        if X.empty:
            raise ValueError("Prediction features must not be empty")
        if self.model is None:
            raise RuntimeError("Le modèle n'est pas entraîné — appeler fit() d'abord.")

        X_seq, _ = self._build_inference_sequences(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_seq).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "feature_columns": self.feature_columns,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "config": {
                    "sequence_length": self.sequence_length,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "random_state": self.random_state,
                },
            },
            path.with_suffix(".pt"),
        )

    @classmethod
    def load(cls, path: Path) -> LSTMModel:
        path = Path(path).with_suffix(".pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        instance = cls(**checkpoint["config"])
        instance.feature_columns = checkpoint["feature_columns"]
        instance.feature_mean = checkpoint["feature_mean"]
        instance.feature_std = checkpoint["feature_std"]

        instance.model = _LSTMNetwork(
            n_features=len(instance.feature_columns),
            hidden_size=instance.hidden_size,
            num_layers=instance.num_layers,
            dropout=instance.dropout,
        ).to(instance.device)
        instance.model.load_state_dict(checkpoint["model_state"])
        instance.model.eval()

        return instance
