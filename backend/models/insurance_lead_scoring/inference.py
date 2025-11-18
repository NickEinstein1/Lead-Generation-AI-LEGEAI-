import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Union
import hashlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class InsuranceLeadScorer:
    """
    Base class for insurance lead scoring.
    Can work with or without ML models - gracefully falls back to rule-based scoring.
    """
    def __init__(self, model_path='models/insurance_lead_scoring/artifacts', require_model=False):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.require_model = require_model
        self.load_model()

    def load_model(self):
        """Load trained model and preprocessors - gracefully handles missing models"""
        try:
            self.model = joblib.load(f'{self.model_path}/model.pkl')
            self.scaler = joblib.load(f'{self.model_path}/scaler.pkl')
            self.label_encoders = joblib.load(f'{self.model_path}/label_encoders.pkl')
            logger.info("Insurance lead scoring model loaded successfully")
        except Exception as e:
            if self.require_model:
                logger.error(f"Error loading required model: {e}")
                raise
            else:
                logger.warning(f"Model not available: {e}. Using rule-based scoring fallback.")
                self.model = None
                self.scaler = None
                self.label_encoders = {}
    
    def anonymize_pii(self, data: Dict) -> Dict:
        """Anonymize PII data for compliance"""
        sensitive_fields = ['email', 'phone', 'ssn']
        anonymized_data = data.copy()
        
        for field in sensitive_fields:
            if field in anonymized_data:
                # Hash sensitive data
                anonymized_data[field] = hashlib.sha256(
                    str(anonymized_data[field]).encode()
                ).hexdigest()[:16]
        
        return anonymized_data
    
    def validate_consent(self, lead_data: Dict) -> bool:
        """Validate lead consent for processing"""
        return lead_data.get('consent_given', False) and \
               lead_data.get('consent_timestamp') is not None
    
    def preprocess_lead(self, lead_data: Dict) -> np.ndarray:
        """Preprocess single lead for scoring"""
        # Convert to DataFrame for consistent processing
        df = pd.DataFrame([lead_data])
        
        # Handle categorical variables
        categorical_cols = ['policy_type', 'previous_insurance']
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Feature engineering
        df['age_income_ratio'] = df['age'] / (df['income'] / 1000)
        df['engagement_per_request'] = df['social_engagement_score'] / (df['quote_requests_30d'] + 1)
        
        # Select features
        feature_columns = [
            'age', 'income', 'policy_type', 'quote_requests_30d',
            'social_engagement_score', 'location_risk_score',
            'previous_insurance', 'credit_score_proxy',
            'age_income_ratio', 'engagement_per_request'
        ]
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Scale features
        X = self.scaler.transform(df[feature_columns])
        
        return X
    
    def score_lead(self, lead_data: Dict) -> Dict:
        """Score a single insurance lead"""
        try:
            # Validate consent
            if not self.validate_consent(lead_data):
                return {
                    'lead_id': lead_data.get('lead_id'),
                    'score': 0,
                    'error': 'Consent not provided or invalid',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            
            # Anonymize PII
            anonymized_data = self.anonymize_pii(lead_data)
            
            # Preprocess
            X = self.preprocess_lead(anonymized_data)
            
            # Predict
            score = self.model.predict(X)[0]
            
            # Ensure score is within 0-100 range
            score = max(0, min(100, score))
            
            return {
                'lead_id': lead_data.get('lead_id'),
                'score': round(float(score), 2),
                'confidence': self._calculate_confidence(X),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'model_version': '1.0'
            }
            
        except Exception as e:
            logger.error(f"Error scoring lead {lead_data.get('lead_id')}: {e}")
            return {
                'lead_id': lead_data.get('lead_id'),
                'score': 0,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _calculate_confidence(self, X: np.ndarray) -> float:
        """Calculate prediction confidence"""
        # Use feature importance and prediction variance as confidence proxy
        try:
            # For XGBoost, we can use prediction with output_margin
            margin = self.model.predict(X, output_margin=True)[0]
            confidence = 1 / (1 + np.exp(-abs(margin)))  # Sigmoid transformation
            return round(float(confidence), 3)
        except:
            return 0.5  # Default confidence
    
    def batch_score(self, leads: List[Dict]) -> List[Dict]:
        """Score multiple leads efficiently"""
        results = []
        for lead in leads:
            result = self.score_lead(lead)
            results.append(result)
        
        return results

# Example usage
if __name__ == "__main__":
    scorer = InsuranceLeadScorer()
    
    sample_lead = {
        'lead_id': 'INS_001',
        'age': 35,
        'income': 75000,
        'policy_type': 'auto',
        'quote_requests_30d': 3,
        'social_engagement_score': 8.5,
        'location_risk_score': 6.2,
        'previous_insurance': 'yes',
        'credit_score_proxy': 720,
        'consent_given': True,
        'consent_timestamp': '2024-01-15T10:30:00Z'
    }
    
    result = scorer.score_lead(sample_lead)
    print(f"Lead Score: {result}")