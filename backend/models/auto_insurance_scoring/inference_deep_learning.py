"""
Auto Insurance Deep Learning Inference

This module provides inference capabilities for the auto insurance deep learning model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import torch
import torch.nn as nn
import joblib
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoInsuranceNeuralNetwork(nn.Module):
    """Deep Learning model for auto insurance (same architecture as training)"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(AutoInsuranceNeuralNetwork, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        
        self.hidden1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        
        self.hidden2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        
        self.attention = nn.Linear(hidden_sizes[2], 1)
        self.output_layer = nn.Linear(hidden_sizes[2], 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Handle single sample case for batch normalization
        single_sample = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        elif x.size(0) == 1 and self.training:
            # If batch size is 1 during training, this will cause issues
            # But during inference (eval mode), batch norm uses running stats
            pass

        x = self.input_layer(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.hidden1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.bn3(x)
        x = self.relu(x)

        attention_weights = self.sigmoid(self.attention(x))
        x = x * attention_weights

        x = self.output_layer(x)
        x = torch.clamp(x, 0, 100)

        if single_sample:
            return x.squeeze()

        return x.squeeze(-1)


class AutoInsuranceDeepLearningScorer:
    """Inference class for auto insurance deep learning model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessors"""
        model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        
        try:
            # Load preprocessors
            self.scaler = joblib.load(os.path.join(model_dir, 'auto_insurance_dl_scaler.pkl'))
            self.label_encoders = joblib.load(os.path.join(model_dir, 'auto_insurance_dl_encoders.pkl'))
            self.feature_columns = joblib.load(os.path.join(model_dir, 'auto_insurance_dl_features.pkl'))
            
            # Load model
            input_size = len(self.feature_columns) + 3  # +3 for engineered features
            self.model = AutoInsuranceNeuralNetwork(input_size).to(self.device)
            self.model.load_state_dict(torch.load(
                os.path.join(model_dir, 'auto_insurance_dl_model.pth'),
                map_location=self.device
            ))
            self.model.eval()
            
            logger.info("Auto insurance deep learning model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load auto insurance deep learning model: {e}")
            raise
    
    def preprocess_lead(self, lead_data: Dict) -> np.ndarray:
        """Preprocess a single lead for prediction"""
        df = pd.DataFrame([lead_data])
        
        # Handle categorical variables
        categorical_cols = ['policy_type', 'previous_insurance']
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Feature engineering
        df['age_income_ratio'] = df['age'] / (df['income'] / 1000)
        df['engagement_per_request'] = df['social_engagement_score'] / (df['quote_requests_30d'] + 1)
        df['risk_engagement_product'] = df['location_risk_score'] * df['social_engagement_score']
        
        # Select features
        feature_df = df[self.feature_columns + ['age_income_ratio', 'engagement_per_request', 'risk_engagement_product']]
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
        
        # Scale
        X_scaled = self.scaler.transform(feature_df)
        
        return X_scaled
    
    def _calculate_confidence(self, X: np.ndarray) -> float:
        """Calculate prediction confidence based on model uncertainty"""
        try:
            # For single samples, we can't use train mode with batch norm
            # So we'll use a simpler confidence estimation
            if X.shape[0] == 1:
                # Use a fixed confidence for single samples
                # In production, this could be based on feature quality
                return 0.85

            # Use dropout at inference time for uncertainty estimation (Monte Carlo dropout)
            # Only for batch predictions
            self.model.train()  # Enable dropout
            predictions = []

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                for _ in range(10):  # Monte Carlo dropout
                    pred = self.model(X_tensor).cpu().numpy()
                    predictions.append(pred)

            self.model.eval()  # Restore eval mode

            # Calculate confidence from prediction variance
            predictions = np.array(predictions)
            variance = np.var(predictions)
            confidence = 1.0 / (1.0 + variance)  # Higher variance = lower confidence

            return min(0.99, max(0.01, float(confidence)))

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}, using default")
            self.model.eval()  # Ensure model is back in eval mode
            return 0.75  # Default confidence

    def score_lead(self, lead_data: Dict) -> Dict[str, Any]:
        """Score a single auto insurance lead"""
        try:
            # Preprocess
            X = self.preprocess_lead(lead_data)

            # Predict - ensure model is in eval mode
            self.model.eval()
            with torch.no_grad():
                # Ensure X has batch dimension
                if X.ndim == 1:
                    X = X.reshape(1, -1)

                X_tensor = torch.FloatTensor(X).to(self.device)
                prediction = self.model(X_tensor).cpu().numpy()

                # Handle both scalar and array outputs
                if prediction.ndim == 0:
                    score = float(prediction.item())
                elif len(prediction.shape) == 0:
                    score = float(prediction)
                else:
                    score = float(prediction[0] if len(prediction) > 0 else prediction)

            # Ensure score is in valid range
            score = max(0, min(100, float(score)))

            # Calculate confidence
            confidence = self._calculate_confidence(X)

            return {
                'lead_id': lead_data.get('lead_id', 'unknown'),
                'score': round(score, 2),
                'confidence': round(confidence, 3),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'model_version': '1.0_auto_insurance_dl',
                'model_type': 'deep_learning'
            }

        except Exception as e:
            logger.error(f"Error scoring auto insurance lead: {e}")
            return {
                'lead_id': lead_data.get('lead_id', 'unknown'),
                'score': 50.0,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def batch_score(self, leads: List[Dict]) -> List[Dict]:
        """Score multiple auto insurance leads"""
        results = []
        for lead in leads:
            result = self.score_lead(lead)
            results.append(result)
        return results


# Example usage
if __name__ == "__main__":
    try:
        scorer = AutoInsuranceDeepLearningScorer()

        sample_lead = {
            'lead_id': 'AUTO_001',
            'age': 35,
            'income': 75000,
            'policy_type': 'auto',
            'quote_requests_30d': 3,
            'social_engagement_score': 8.5,
            'location_risk_score': 6.2,
            'previous_insurance': 'yes',
            'credit_score_proxy': 720,
            'consent_given': True,
            'consent_timestamp': '2024-11-19T10:30:00Z'
        }

        result = scorer.score_lead(sample_lead)

        print("\n" + "=" * 80)
        print("AUTO INSURANCE DEEP LEARNING SCORING RESULT")
        print("=" * 80)
        print(f"Lead ID: {result['lead_id']}")
        print(f"Score: {result['score']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Model Type: {result.get('model_type', 'N/A')}")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Failed to run example: {e}")
        logger.info("Model files not found. Train the model first using train_deep_learning.py")

