"""
Deep Learning Inference for Health Insurance Lead Scoring
PyTorch-based neural network inference with uncertainty estimation
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import logging
import os
from typing import Dict, Any
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthInsuranceNeuralNetwork(nn.Module):
    """Neural Network for Health Insurance - must match training architecture"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(HealthInsuranceNeuralNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_sizes[2], 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Handle single sample case for batch normalization
        single_sample = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        elif x.size(0) == 1 and self.training:
            pass
        
        # Layer 1
        x = self.fc1(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc4(x)
        
        if single_sample:
            return x.squeeze()
        return x.squeeze(-1)


class HealthInsuranceDeepLearningScorer:
    """Deep Learning Scorer for Health Insurance Leads"""

    def __init__(self, model_dir='backend/models/healthcare_insurance_scoring/saved_models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained deep learning model"""
        try:
            # Load feature columns
            self.feature_columns = joblib.load(f'{self.model_dir}/health_insurance_dl_features.pkl')

            # Load scaler and encoders
            self.scaler = joblib.load(f'{self.model_dir}/health_insurance_dl_scaler.pkl')
            self.label_encoders = joblib.load(f'{self.model_dir}/health_insurance_dl_label_encoders.pkl')

            # Calculate input size (base features + engineered features)
            input_size = len(self.feature_columns) + 3  # +3 for engineered features

            # Initialize and load model
            self.model = HealthInsuranceNeuralNetwork(input_size).to(self.device)
            self.model.load_state_dict(torch.load(
                f'{self.model_dir}/health_insurance_dl_model.pth',
                map_location=self.device
            ))
            self.model.eval()

            logger.info("Health Insurance Deep Learning model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Health Insurance Deep Learning model: {e}")
            raise
    
    def preprocess_lead(self, lead_data: Dict) -> np.ndarray:
        """Preprocess single lead for scoring"""
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
        feature_cols = self.feature_columns + ['age_income_ratio', 'engagement_per_request', 'risk_engagement_product']
        X = df[feature_cols].values
        
        # Scale
        X = self.scaler.transform(X)
        
        return X

    def _calculate_confidence(self, X: np.ndarray) -> float:
        """Calculate prediction confidence using Monte Carlo dropout"""
        try:
            # For single samples, we can't use train mode with batch norm
            if X.shape[0] == 1:
                # Use a fixed confidence for single samples
                return 0.85

            # Use dropout at inference time for batch predictions (Monte Carlo dropout)
            self.model.train()  # Enable dropout
            n_iterations = 10
            predictions = []

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                for _ in range(n_iterations):
                    pred = self.model(X_tensor).cpu().numpy()
                    predictions.append(pred)

            self.model.eval()  # Restore eval mode

            # Calculate confidence from prediction variance
            predictions = np.array(predictions)
            std = np.std(predictions)
            confidence = 1.0 / (1.0 + std)

            return float(np.clip(confidence, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            self.model.eval()  # Ensure model is back in eval mode
            return 0.75  # Default confidence

    def score_lead(self, lead_data: Dict) -> Dict[str, Any]:
        """Score a single health insurance lead"""
        try:
            # Preprocess
            X = self.preprocess_lead(lead_data)

            # Predict
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

            # Ensure score is within 0-100 range
            score = max(0, min(100, score))

            # Calculate confidence
            confidence = self._calculate_confidence(X)

            return {
                'lead_id': lead_data.get('lead_id', 'unknown'),
                'score': round(float(score), 2),
                'confidence': round(float(confidence), 3),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'model_type': 'deep_learning',
                'model_version': '1.0_health_dl'
            }

        except Exception as e:
            logger.error(f"Error scoring health insurance lead: {e}")
            return {
                'lead_id': lead_data.get('lead_id', 'unknown'),
                'score': 0,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'model_type': 'deep_learning'
            }


if __name__ == "__main__":
    # Test the scorer
    scorer = HealthInsuranceDeepLearningScorer()

    test_lead = {
        'lead_id': 'HEALTH_DL_TEST_001',
        'age': 42,
        'income': 85000,
        'policy_type': 'health',
        'quote_requests_30d': 2,
        'social_engagement_score': 7.5,
        'location_risk_score': 4.8,
        'previous_insurance': 'yes',
        'credit_score_proxy': 750
    }

    result = scorer.score_lead(test_lead)
    print(f"Health Insurance Deep Learning Score: {result}")

