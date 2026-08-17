import torch
import torch.nn as nn
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR


class BatteryFailureNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class BatteryPredictor:
    def __init__(self):
        with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
            self.metadata = json.load(f)

        self.feature_columns = self.metadata["feature_columns"]
        self.cat_cols = self.metadata["cat_cols"]
        self.num_cols = self.metadata["num_cols"]
        self.threshold = self.metadata["threshold"]
        input_dim = self.metadata["input_dim"]

        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

        self.model = BatteryFailureNet(input_dim)
        self.model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "battery_model.pth"), map_location="cpu")
        )
        self.model.eval()

    def _encode(self, data: dict) -> np.ndarray:
        row = []
        for col in self.feature_columns:
            val = data.get(col)
            if col in self.cat_cols:
                le = self.encoders[col]
                val = str(val) if val is not None else "missing"
                if val not in le.classes_:
                    val = "missing" if "missing" in le.classes_ else le.classes_[0]
                val = le.transform([val])[0]
            else:
                val = float(val) if val is not None else 0.0
            row.append(val)
        return np.array(row, dtype=np.float32).reshape(1, -1)

    def predict(self, data: dict) -> dict:
        raw = self._encode(data)
        scaled = self.scaler.transform(raw)
        x = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            logit = self.model(x).squeeze(1)
            prob = torch.sigmoid(logit).item()

        label = "Failure" if prob >= self.threshold else "No Failure"
        return {
            "probability_failure": round(prob, 4),
            "prediction": label,
            "threshold_used": self.threshold
        }


predictor = BatteryPredictor()
