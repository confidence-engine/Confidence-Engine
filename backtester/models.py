import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class EnhancedMLP(nn.Module):
    """Enhanced MLP with batch normalization and dropout"""
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout_rate: float = 0.2):
        super().__init__()
        layers = []

        # Input layer
        layers.extend([
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AttentionMLP(nn.Module):
    """MLP with attention mechanism for feature importance"""
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout_rate: float = 0.2):
        super().__init__()
        self.feature_projection = nn.Linear(in_dim, hidden_dims[0])
        self.attention = nn.MultiheadAttention(hidden_dims[0], num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dims[0])
        self.dropout = nn.Dropout(dropout_rate)

        # MLP layers after attention
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_dim) or (batch_size, in_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if needed

        # Project features
        proj = self.feature_projection(x)

        # Apply attention
        attn_out, _ = self.attention(proj, proj, proj)
        attn_out = self.norm(attn_out + proj)  # Residual connection
        attn_out = self.dropout(attn_out)

        # Global average pooling across sequence dimension
        pooled = torch.mean(attn_out, dim=1)

        # MLP prediction
        return self.mlp(pooled)


class LSTMModel(nn.Module):
    """LSTM-based model for sequential data"""
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if needed

        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # Take last layer's hidden state
        dropped = self.dropout(last_hidden)
        return self.fc(dropped)


class TransformerModel(nn.Module):
    """Transformer-based model for sequential data"""
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_heads: int = 8, dropout_rate: float = 0.2):
        super().__init__()
        self.input_projection = nn.Linear(in_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))  # Max sequence length 1000

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if needed

        # Add positional encoding
        seq_len = x.size(1)
        x = self.input_projection(x) + self.pos_encoding[:, :seq_len, :]

        # Apply transformer
        transformer_out = self.transformer(x)

        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        dropped = self.dropout(pooled)
        return self.fc(dropped)


class CNNModel(nn.Module):
    """CNN-based model for pattern recognition"""
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 4, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if needed

        # Transpose for Conv1d: (batch, in_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Global average pooling
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class EnsembleModel(nn.Module):
    """Ensemble of multiple model architectures"""
    def __init__(self, in_dim: int, hidden_dims: List[int] = [64, 32], dropout_rate: float = 0.2):
        super().__init__()
        self.models = nn.ModuleList([
            EnhancedMLP(in_dim, hidden_dims, dropout_rate),
            AttentionMLP(in_dim, hidden_dims, dropout_rate),
            LSTMModel(in_dim, hidden_dims[0]),
            TransformerModel(in_dim, hidden_dims[0]),
            CNNModel(in_dim, hidden_dims[0], dropout_rate)
        ])

        # Ensemble weights (learned)
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)))
        self.final_fc = nn.Linear(len(self.models), 1)

    def forward(self, x):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            try:
                pred = model(x)
                predictions.append(pred)
            except Exception:
                # If a model fails, use zero prediction
                predictions.append(torch.zeros(x.size(0), 1, device=x.device))

        # Stack predictions
        stacked = torch.stack(predictions, dim=-1).squeeze(1)  # (batch_size, num_models)

        # Apply ensemble weights
        weighted = stacked * F.softmax(self.ensemble_weights, dim=0)

        # Final prediction
        return self.final_fc(weighted)
