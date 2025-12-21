import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from pathlib import Path
import json

class ModelInference:
    def __init__(self, model_dir=None, model_version="v4"):
        """
        Initialize ModelInference.
        
        Args:
            model_dir (str or Path, optional): Directory containing model files. Defaults to current directory.
            model_version (str, optional): 'v4' (Main) or 'v5' (Behavior). Defaults to 'v4'.
        """
        if model_dir is None:
            # Default: Current directory (03_trained_model)
            model_dir = Path(__file__).parent
        
        self.model_dir = Path(model_dir)
        self.model_version = model_version.lower()

        # Select Model Files based on version
        if self.model_version == "v4":
            self.model_filename = "catboost_model_v4.cbm"
            self.metadata_filename = "model_metadata_v4.json"
        elif self.model_version == "v5":
            self.model_filename = "catboost_model_v5_behavior.cbm"
            self.metadata_filename = "model_metadata_v5_behavior.json"
        elif self.model_version == "v5.2":
            self.model_filename = "catboost_model_v5.2.cbm"
            self.metadata_filename = "model_metadata_v5.2.json"
        else:
            raise ValueError(f"Unknown model version: {model_version}. Options: 'v4', 'v5', 'v5.2'")

        self.model_path = self.model_dir / self.model_filename
        self.metadata_path = self.model_dir / self.metadata_filename
        
        # Load metadata
        if not self.metadata_path.exists():
             raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.feature_names = self.metadata["feature_names"]
        self.categorical_cols = self.metadata.get("categorical_features", [])
        
        # Load Model
        if not self.model_path.exists():
             raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = CatBoostClassifier()
        self.model.load_model(str(self.model_path))
        print(f"Loaded Model: {self.model_version} ({self.metadata.get('description', 'No description')})")
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate V4 strategic derived features from V3 base features.
        Includes outlier handling and robust calculations.
        
        Note: V5 model might not use all of these, but generating them is safe 
        as the model will only select what it needs in 'preprocess'.
        """
        df_eng = df.copy()
        epsilon = 1e-6
        
        # 1. Active Decay Rate
        if "num_days_active_w7" in df_eng.columns and "num_days_active_w30" in df_eng.columns:
            val = df_eng['num_days_active_w7'] / ((df_eng['num_days_active_w30'] / 4) + epsilon)
            df_eng['active_decay_rate'] = val.clip(upper=10.0) 
            
        # 2. Listening Time Velocity
        if "avg_secs_per_day_w7" in df_eng.columns and "avg_secs_per_day_w14" in df_eng.columns:
            df_eng['listening_time_velocity'] = df_eng['avg_secs_per_day_w7'] - df_eng['avg_secs_per_day_w14']
            
        # 3. Discovery Index
        if "num_unq_w7" in df_eng.columns and "num_songs_w7" in df_eng.columns:
            val = df_eng['num_unq_w7'] / (df_eng['num_songs_w7'] + epsilon)
            df_eng['discovery_index'] = val.clip(upper=1.0)
            
        # 4. Skip Passion Index
        if "num_25_w7" in df_eng.columns and "num_100_w7" in df_eng.columns:
            val = df_eng['num_25_w7'] / (df_eng['num_100_w7'] + epsilon)
            df_eng['skip_passion_index'] = val.clip(upper=100.0)
            
        # 5. Daily Listening Variance (Renaming)
        if "std_secs_w7" in df_eng.columns:
            df_eng['daily_listening_variance'] = df_eng['std_secs_w7']
            
        # 6. Engagement Density
        if "total_secs_w7" in df_eng.columns and "num_days_active_w7" in df_eng.columns:
            val = df_eng['total_secs_w7'] / (df_eng['num_days_active_w7'] + epsilon)
            df_eng['engagement_density'] = val

        # 7. Last Active Gap
        if "last_active_gap" not in df_eng.columns:
            df_eng['last_active_gap'] = -1 
            
        # 8. Weighted Behavioral Features (V5) - Optional calculation
        # Even if V4 doesn't use them, we can calculate if columns exist
        if "avg_secs_per_day_w7" in df_eng.columns:
             # Just a placeholder if we need more complex logic later
             pass

        return df_eng

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input dataframe for prediction.
        - Generates derived features.
        - Selects features in training order.
        - Ensures categorical columns are strings/categories.
        """
        # Feature Engineering 
        df_eng = self.engineer_features(df)
        
        # Select Only features used in training, in correct order
        # This automatically handles V4 vs V5 difference (V5 drops many columns)
        try:
            # Check for missing columns
            missing = list(set(self.feature_names) - set(df_eng.columns))
            if missing:
                # Special handling: If V5 uses new features not in input, we might need to calc them
                # But currently V5 uses subset of V4 features (mostly), or basic ones present in raw
                raise KeyError(f"Input dataframe is missing features required for {self.model_version}: {missing}")
                
            df_processed = df_eng[self.feature_names].copy()
            
        except KeyError as e:
            raise e
        
        # Ensure categorical columns are strictly string
        for col in self.categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str).astype("category")
                
        return df_processed
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probability.
        Returns array of probabilities for class 1 (Churn).
        """
        df_processed = self.preprocess(df)
        return self.model.predict_proba(df_processed)[:, 1]

    def predict_label(self, df: pd.DataFrame, threshold=0.5) -> np.ndarray:
        """
        Predict churn label based on threshold.
        """
        probs = self.predict(df)
        return (probs >= threshold).astype(int)

if __name__ == "__main__":
    # Example usage
    try:
        print("Testing V4 Model...")
        inferencer_v4 = ModelInference(model_version="v4")
        
        print("\nTesting V5 Model...")
        inferencer_v5 = ModelInference(model_version="v5")

        print("\nTesting V5.2 Model...")
        inferencer_v5_2 = ModelInference(model_version="v5.2")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
