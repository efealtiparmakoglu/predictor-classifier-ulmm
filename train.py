#!/usr/bin/env python3
"""
Predictor Classifier Ulmm - Machine Learning Pipeline
Complete ML workflow with training and inference
Generated: 2026-04-08 06:19:19
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPipeline:
    """Complete ML training pipeline"""
    
    def __init__(self, config_path='config/model_config.json'):
        self.config = self.load_config(config_path)
        self.models = {}
        self.scaler = StandardScaler()
        self.metrics = {}
        
    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.default_config()
    
    def default_config(self):
        return {
            'test_size': 0.2,
            'random_state': 42,
            'models': ['random_forest', 'gradient_boosting'],
            'cv_folds': 5
        }
    
    def load_data(self, data_path):
        """Load and validate training data"""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Basic validation
        assert not df.empty, "Dataset is empty"
        assert 'target' in df.columns, "Target column missing"
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        return X, y
    
    def preprocess(self, X_train, X_test):
        """Preprocess features"""
        logger.info("Preprocessing features")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        logger.info("Training models")
        
        models_config = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        for name, model in models_config.items():
            logger.info(f"Training {name}")
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.config['cv_folds'])
            self.metrics[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            logger.info(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def evaluate(self, X_test, y_test):
        """Evaluate all models"""
        logger.info("Evaluating models")
        
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            logger.info(f"{name} Accuracy: {results[name]['accuracy']:.4f}")
        
        return results
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            path = f"{output_dir}/{name}_{datetime.now().strftime('%Y%m%d')}.joblib"
            joblib.dump(model, path)
            logger.info(f"Saved {name} to {path}")
        
        # Save scaler
        joblib.dump(self.scaler, f"{output_dir}/scaler.joblib")
    
    def run(self, data_path):
        """Run complete pipeline"""
        logger.info("Starting ML pipeline")
        
        # Load data
        X, y = self.load_data(data_path)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Preprocess
        X_train_scaled, X_test_scaled = self.preprocess(X_train, X_test)
        
        # Train
        self.train_models(X_train_scaled, y_train)
        
        # Evaluate
        results = self.evaluate(X_test_scaled, y_test)
        
        # Save
        self.save_models()
        
        logger.info("Pipeline completed")
        return results

if __name__ == '__main__':
    pipeline = MLPipeline()
    # pipeline.run('data/train.csv')  # Uncomment with actual data
    print("ML Pipeline ready. Provide training data to begin.")
