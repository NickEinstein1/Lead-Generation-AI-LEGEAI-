"""
Deep Learning Inference for Life Insurance Lead Scoring

This module provides inference capabilities for the deep learning models
trained for life insurance lead scoring.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import numpy as np
import pandas as pd
import joblib
import logging
import json
from typing import Dict, Any
from datetime import datetime

from backend.models.life_insurance_scoring.train_deep_learning import LifeInsuranceDeepNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LifeInsuranceDeepLearningScorer:
    """Deep Learning scorer for life insurance leads"""
    
    def __init__(self, model_path='models/life_insurance_scoring/deep_learning_artifacts'):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.config = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained deep learning model and preprocessors"""
        try:
            # Load config
            config_path = f'{self.model_path}/config.json'
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Load scaler and encoders
            self.scaler = joblib.load(f'{self.model_path}/scaler.pkl')
            self.label_encoders = joblib.load(f'{self.model_path}/label_encoders.pkl')
            
            # Initialize and load model
            self.model = LifeInsuranceDeepNetwork(
                input_dim=self.config['input_dim'],
                hidden_dims=[256, 128, 64, 32],
                dropout_rate=0.3,
                use_attention=True
            ).to(self.device)
            
            model_state_path = f'{self.model_path}/best_model.pth'
            self.model.load_state_dict(torch.load(model_state_path, map_location=self.device))
            self.model.eval()
            
            logger.info(f"Deep learning model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading deep learning model: {e}")
            raise
    
    def preprocess_lead(self, lead_data: Dict) -> np.ndarray:
        """Preprocess a single lead for inference"""
        
        # Convert to DataFrame
        df = pd.DataFrame([lead_data])
        
        # Handle categorical variables
        categorical_cols = ['marital_status', 'employment_status', 'health_status', 
                           'smoking_status', 'education_level', 'occupation_risk_level', 'life_stage']
        
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Feature engineering (same as training)
        df['age_risk_factor'] = np.where(df['age'] > 60, df['age'] * 1.5, 
                                np.where(df['age'] > 45, df['age'] * 1.2, df['age']))
        
        df['coverage_income_ratio'] = df['coverage_amount_requested'] / (df['income'] + 1)

        # Handle missing columns with defaults
        mortgage_balance = df['mortgage_balance'] if 'mortgage_balance' in df.columns else 0
        debt_obligations = df['debt_obligations'] if 'debt_obligations' in df.columns else 0
        occupation_risk = df['occupation_risk_level'] if 'occupation_risk_level' in df.columns else 0

        df['financial_responsibility_score'] = (
            df['dependents_count'] * 2 +
            mortgage_balance / 100000 +
            debt_obligations / 50000
        )

        df['mortality_risk_score'] = (
            (df['age'] / 10) +
            (df['smoking_status'] * 3) +
            (df['health_status'] * 2) +
            (occupation_risk * 1.5)
        )
        
        # Select features
        feature_cols = self.config['feature_columns'] + [
            'age_risk_factor', 'coverage_income_ratio', 'financial_responsibility_score',
            'mortality_risk_score'
        ]
        
        feature_df = df[feature_cols]
        feature_df = feature_df.fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(feature_df.values)
        
        return features_scaled
    
    def score_lead(self, lead_data: Dict) -> Dict[str, Any]:
        """Score a life insurance lead using deep learning model"""
        
        try:
            # Preprocess
            X = self.preprocess_lead(lead_data)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Predict
            with torch.no_grad():
                score = self.model(X_tensor)
                # Handle both scalar and array outputs
                score_np = score.cpu().numpy()
                score_value = float(score_np.item() if score_np.ndim == 0 else score_np[0])
            
            # Ensure score is in valid range
            score_value = max(0, min(100, score_value))
            
            # Calculate confidence (based on score proximity to extremes)
            confidence = 1.0 - abs(score_value - 50) / 50.0
            confidence = max(0.5, min(0.95, confidence))
            
            return {
                'lead_id': lead_data.get('lead_id', 'unknown'),
                'score': round(score_value, 2),
                'confidence': round(confidence, 3),
                'model_type': 'deep_learning',
                'model_architecture': 'attention_dnn',
                'device': str(self.device),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error scoring lead with deep learning: {e}")
            return {
                'lead_id': lead_data.get('lead_id', 'unknown'),
                'score': 50.0,
                'error': str(e),
                'model_type': 'deep_learning',
                'timestamp': datetime.utcnow().isoformat()
            }

