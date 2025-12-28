import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

class PNMLRModel:
    def __init__(self, model_path: Path = None):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42)
        self.is_trained = False
        if model_path and model_path.exists():
            try:
                data = joblib.load(model_path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True
            except: pass

    def predict_score(self, features: pd.DataFrame) -> np.ndarray:
        required = ['dist_to_school', 'dist_to_hospital', 'dist_to_park']
        for col in required:
            if col not in features.columns: features[col] = 5000.0
        df_clean = features[required].fillna(5000.0)
        if not self.is_trained: return self._heuristic_score(df_clean)
        X_scaled = self.scaler.transform(df_clean)
        return self.model.predict(X_scaled)

    def _heuristic_score(self, df):
        # RELAXED DECAY for "Aggressive 100" capability
        # Standard urban planning: 15-minute city (~1.5km radius)
        s_school = 100 * np.exp(-df['dist_to_school'] / 1500) 
        s_hosp = 100 * np.exp(-df['dist_to_hospital'] / 2500) # Hospitals can be far
        s_park = 100 * np.exp(-df['dist_to_park'] / 1000)     # Parks closer
        
        final_score = (0.4 * s_school) + (0.3 * s_hosp) + (0.3 * s_park)
        return np.clip(final_score, 0, 100)