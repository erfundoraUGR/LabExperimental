import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=(77, 57), dropout=0.0):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes debe tener al menos una capa.")

        layers = []
        prev_size = input_size
        for i, h_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            # Dropout opcional entre capas ocultas para regularizar.
            if dropout > 0 and i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_size = h_size

        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=64, dropout=0.0):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes debe tener al menos una capa.")

        self.lstm_layers = nn.ModuleList()
        prev_size = input_size
        for h_size in hidden_sizes:
            self.lstm_layers.append(
                nn.LSTM(prev_size, h_size, num_layers=1, batch_first=True)
            )
            prev_size = h_size

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        out = x
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(out)
            if self.dropout is not None and i < len(self.lstm_layers) - 1:
                out = self.dropout(out)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=64, dropout=0.0):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes debe tener al menos una capa.")

        self.gru_layers = nn.ModuleList()
        prev_size = input_size
        for h_size in hidden_sizes:
            self.gru_layers.append(
                nn.GRU(prev_size, h_size, num_layers=1, batch_first=True)
            )
            prev_size = h_size

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        out = x
        for i, gru in enumerate(self.gru_layers):
            out, _ = gru(out)
            if self.dropout is not None and i < len(self.gru_layers) - 1:
                out = self.dropout(out)
        return self.fc(out[:, -1, :])


class ElmanModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_sizes=64,
        dropout=0.0,
        nonlinearity="tanh",
    ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes debe tener al menos una capa.")

        self.rnn_layers = nn.ModuleList()
        prev_size = input_size
        for h_size in hidden_sizes:
            self.rnn_layers.append(
                nn.RNN(
                    prev_size,
                    h_size,
                    num_layers=1,
                    batch_first=True,
                    nonlinearity=nonlinearity,
                )
            )
            prev_size = h_size

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        out = x
        for i, rnn in enumerate(self.rnn_layers):
            out, _ = rnn(out)
            if self.dropout is not None and i < len(self.rnn_layers) - 1:
                out = self.dropout(out)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_sizes=64,
        dropout=0.0,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes debe tener al menos una capa.")

        # La primera capa es la dimensión del transformer (d_model)
        d_model = hidden_sizes[0]

        if d_model == 0 or d_model < num_heads:
            raise ValueError(
                f"hidden_size ({d_model}) debe ser >= num_heads ({num_heads})"
            )

        if d_model % num_heads != 0:
            raise ValueError(
                f"hidden_size ({d_model}) debe ser divisible por num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads

        # Embedding layer
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoding = self._get_positional_encoding(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Capas lineales adicionales (si hay más de una capa oculta)
        linear_layers = []
        prev_size = d_model
        for i, h_size in enumerate(hidden_sizes[1:], start=1):
            linear_layers.append(nn.Linear(prev_size, h_size))
            linear_layers.append(nn.ReLU())
            if dropout > 0 and i < len(hidden_sizes) - 1:
                linear_layers.append(nn.Dropout(dropout))
            prev_size = h_size

        if len(hidden_sizes) > 1:
            # Si hay capas adicionales, agregar la capa final
            linear_layers.append(nn.Linear(prev_size, 1))
            self.fc_layers = nn.Sequential(*linear_layers)
        else:
            # Si solo hay una capa, usar directamente
            self.fc_layers = nn.Linear(d_model, 1)

    def _get_positional_encoding(self, d_model, max_len=5000):
        """Genera positional encoding sinusoidal."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x shape: (batch_size, seq_len) o (batch_size, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, d_model)

        # Positional encoding
        seq_len = x.size(1)
        device = x.device
        pe = self.pos_encoding[:seq_len, :].to(device)
        x = x + pe.unsqueeze(0)

        # Dropout
        x = self.dropout_layer(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # Tomar el último output
        x = x[:, -1, :]  # (batch_size, d_model)

        # Capas lineales
        x = self.fc_layers(x)  # (batch_size, 1)

        return x



def build_model(config):
    """
    Crea el modelo usando la configuracion actual.
    Centralizar este bloque evita duplicar logica entre `main.py` y `tune_optuna.py`.
    """
    model_type = config.MODEL_TYPE.upper()

    if model_type == "MLP":
        hidden_sizes = getattr(config, "MLP_HIDDEN_SIZES", [77, 57])
        dropout = getattr(config, "MLP_DROPOUT", 0.0)
        return MLP(
            input_size=config.WINDOW_SIZE,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    if model_type == "LSTM":
        if hasattr(config, "LSTM_HIDDEN_SIZES"):
            hidden_sizes = config.LSTM_HIDDEN_SIZES
        else:
            hidden_size = getattr(config, "LSTM_HIDDEN_SIZE", 64)
            num_layers = getattr(config, "LSTM_NUM_LAYERS", 1)
            hidden_sizes = [hidden_size] * num_layers

        return LSTMModel(
            input_size=1,
            hidden_sizes=hidden_sizes,
            dropout=getattr(config, "LSTM_DROPOUT", 0.0),
        )

    if model_type == "GRU":
        if hasattr(config, "GRU_HIDDEN_SIZES"):
            hidden_sizes = config.GRU_HIDDEN_SIZES
        else:
            hidden_size = getattr(config, "GRU_HIDDEN_SIZE", 64)
            num_layers = getattr(config, "GRU_NUM_LAYERS", 1)
            hidden_sizes = [hidden_size] * num_layers

        return GRUModel(
            input_size=1,
            hidden_sizes=hidden_sizes,
            dropout=getattr(config, "GRU_DROPOUT", 0.0),
        )

    if model_type == "ELMAN":
        if hasattr(config, "ELMAN_HIDDEN_SIZES"):
            hidden_sizes = config.ELMAN_HIDDEN_SIZES
        else:
            hidden_size = getattr(config, "ELMAN_HIDDEN_SIZE", 64)
            num_layers = getattr(config, "ELMAN_NUM_LAYERS", 1)
            hidden_sizes = [hidden_size] * num_layers

        return ElmanModel(
            input_size=1,
            hidden_sizes=hidden_sizes,
            dropout=getattr(config, "ELMAN_DROPOUT", 0.0),
            nonlinearity=getattr(config, "ELMAN_NONLINEARITY", "tanh"),
        )

    if model_type == "TRANSFORMER":
        if hasattr(config, "TRANSFORMER_HIDDEN_SIZES"):
            hidden_sizes = config.TRANSFORMER_HIDDEN_SIZES
        else:
            hidden_size = getattr(config, "TRANSFORMER_HIDDEN_SIZE", 64)
            hidden_sizes = [hidden_size]

        dropout = getattr(config, "TRANSFORMER_DROPOUT", 0.0)
        num_heads = getattr(config, "TRANSFORMER_NUM_HEADS", 4)
        num_layers = getattr(config, "TRANSFORMER_NUM_LAYERS", 2)

        return TransformerModel(
            input_size=1,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    raise ValueError(
        "MODEL_TYPE invalido. Use uno de: MLP, LSTM, GRU, ELMAN, TRANSFORMER"
    )
