#!/usr/bin/env python3
"""
Inference script for trained models
"""

import joblib
import numpy as np
import json
from pathlib import Path

class Predictor:
    """Model inference class"""
    
    def __init__(self, model_path='models/random_forest.joblib'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('models/scaler.joblib')
    
    def predict(self, features):
        """Make prediction on new data"""
        if isinstance(features, list):
            features = np.array(features)
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return {
            'prediction': prediction[0],
            'confidence': float(np.max(probabilities)),
            'probabilities': probabilities[0].tolist()
        }
    
    def batch_predict(self, data_path):
        """Batch prediction from CSV"""
        import pandas as pd
        df = pd.read_csv(data_path)
        
        results = []
        for _, row in df.iterrows():
            result = self.predict(row.values)
            results.append(result)
        
        return results

if __name__ == '__main__':
    predictor = Predictor()
    # Example usage
    print("Predictor loaded. Use predictor.predict(features) to make predictions.")
