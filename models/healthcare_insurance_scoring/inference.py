import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Union
import hashlib
from datetime import datetime, date
import sys
sys.path.append('../../')
from models.insurance_lead_scoring.inference import InsuranceLeadScorer

logger = logging.getLogger(__name__)

class HealthcareInsuranceLeadScorer(InsuranceLeadScorer):
    def __init__(self, model_path='models/healthcare_insurance_scoring/artifacts'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.load_model()
        
    def anonymize_pii(self, data: Dict) -> Dict:
        """Enhanced PII anonymization for HIPAA compliance"""
        sensitive_fields = [
            'email', 'phone', 'ssn', 'medical_record_number',
            'health_conditions', 'prescription_names', 'doctor_names'
        ]
        anonymized_data = data.copy()
        
        for field in sensitive_fields:
            if field in anonymized_data:
                # Hash sensitive healthcare data
                anonymized_data[field] = hashlib.sha256(
                    str(anonymized_data[field]).encode()
                ).hexdigest()[:16]
        
        return anonymized_data
    
    def validate_consent(self, lead_data: Dict) -> bool:
        """Validate HIPAA consent for healthcare data processing"""
        return (lead_data.get('hipaa_consent_given', False) and 
                lead_data.get('consent_given', False) and
                lead_data.get('hipaa_consent_timestamp') is not None)
    
    def calculate_open_enrollment_urgency(self, lead_data: Dict) -> float:
        """Calculate urgency based on open enrollment periods"""
        current_date = datetime.now()
        
        # General open enrollment: Nov 1 - Dec 15
        if current_date.month == 11 or (current_date.month == 12 and current_date.day <= 15):
            return 10.0
        
        # Special enrollment period indicators
        if lead_data.get('qualifying_life_event', False):
            return 9.0
            
        # Medicaid/CHIP enrollment (year-round)
        if lead_data.get('subsidy_eligible', False) and lead_data.get('income', 0) < 30000:
            return 8.0
            
        return 3.0  # Default urgency
    
    def preprocess_lead(self, lead_data: Dict) -> np.ndarray:
        """Healthcare-specific lead preprocessing"""
        # Convert to DataFrame
        df = pd.DataFrame([lead_data])
        
        # Handle categorical variables
        categorical_cols = ['employment_status', 'current_coverage']
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Healthcare feature engineering
        df['age_risk_factor'] = np.where(df['age'] > 50, df['age'] * 1.2, df['age'])
        df['health_complexity_score'] = (
            df.get('health_conditions_count', 0) * 2 + 
            df.get('prescription_usage', 0) + 
            df.get('doctor_visits_annual', 2) / 12
        )
        df['affordability_ratio'] = df['income'] / (df.get('family_size', 1) * 12000)
        df['urgency_score'] = self.calculate_open_enrollment_urgency(lead_data)
        df['subsidy_need_score'] = np.where(
            df.get('subsidy_eligible', 0) == 1,
            (400 - (df['income'] / 100)) / 100,
            0
        )
        
        # Feature columns
        feature_columns = [
            'age', 'income', 'family_size', 'employment_status',
            'current_coverage', 'health_conditions_count', 'prescription_usage',
            'doctor_visits_annual', 'preventive_care_usage', 'location_risk_score',
            'open_enrollment_period', 'subsidy_eligible', 'previous_claims_frequency',
            'age_risk_factor', 'health_complexity_score', 'affordability_ratio',
            'urgency_score', 'subsidy_need_score'
        ]
        
        # Handle missing values with healthcare defaults
        df['prescription_usage'] = df['prescription_usage'].fillna(0)
        df['doctor_visits_annual'] = df['doctor_visits_annual'].fillna(2)
        df['health_conditions_count'] = df['health_conditions_count'].fillna(0)
        df['preventive_care_usage'] = df['preventive_care_usage'].fillna(1)
        df = df.fillna(df.median())
        
        # Scale features
        X = self.scaler.transform(df[feature_columns])
        
        return X
    
    def score_lead(self, lead_data: Dict) -> Dict:
        """Score healthcare insurance lead with enhanced logic"""
        try:
            # Validate HIPAA consent
            if not self.validate_consent(lead_data):
                return {
                    'lead_id': lead_data.get('lead_id'),
                    'score': 0,
                    'error': 'HIPAA consent not provided or invalid',
                    'timestamp': datetime.utcnow().isoformat(),
                    'compliance_status': 'FAILED'
                }
            
            # Anonymize PII
            anonymized_data = self.anonymize_pii(lead_data)
            
            # Preprocess
            X = self.preprocess_lead(anonymized_data)
            
            # Predict base score
            base_score = self.model.predict(X)[0]
            
            # Healthcare-specific score adjustments
            adjusted_score = self._apply_healthcare_adjustments(base_score, lead_data)
            
            # Ensure score is within 0-100 range
            final_score = max(0, min(100, adjusted_score))
            
            return {
                'lead_id': lead_data.get('lead_id'),
                'score': round(float(final_score), 2),
                'base_score': round(float(base_score), 2),
                'confidence': self._calculate_confidence(X),
                'urgency_level': self._get_urgency_level(lead_data),
                'recommended_plan_type': self._recommend_plan_type(lead_data),
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': '1.0_healthcare',
                'compliance_status': 'PASSED'
            }
            
        except Exception as e:
            logger.error(f"Error scoring healthcare lead {lead_data.get('lead_id')}: {e}")
            return {
                'lead_id': lead_data.get('lead_id'),
                'score': 0,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'compliance_status': 'ERROR'
            }
    
    def _apply_healthcare_adjustments(self, base_score: float, lead_data: Dict) -> float:
        """Apply healthcare-specific score adjustments"""
        adjusted_score = base_score
        
        # Urgency boost during open enrollment
        if lead_data.get('open_enrollment_period', False):
            adjusted_score *= 1.15
        
        # High health complexity boost
        health_conditions = lead_data.get('health_conditions_count', 0)
        if health_conditions > 2:
            adjusted_score *= 1.1
        
        # Subsidy eligibility boost
        if lead_data.get('subsidy_eligible', False):
            adjusted_score *= 1.08
        
        # No current coverage urgency
        if lead_data.get('current_coverage') == 'none':
            adjusted_score *= 1.12
        
        return adjusted_score
    
    def _get_urgency_level(self, lead_data: Dict) -> str:
        """Determine urgency level for healthcare lead"""
        urgency_score = self.calculate_open_enrollment_urgency(lead_data)
        
        if urgency_score >= 9:
            return 'CRITICAL'
        elif urgency_score >= 7:
            return 'HIGH'
        elif urgency_score >= 5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _recommend_plan_type(self, lead_data: Dict) -> str:
        """Recommend healthcare plan type based on lead profile"""
        age = lead_data.get('age', 30)
        income = lead_data.get('income', 50000)
        health_conditions = lead_data.get('health_conditions_count', 0)
        family_size = lead_data.get('family_size', 1)
        
        # High-deductible health plan for young, healthy individuals
        if age < 30 and health_conditions == 0 and income > 40000:
            return 'HDHP'
        
        # PPO for those with health conditions or higher income
        elif health_conditions > 1 or income > 75000:
            return 'PPO'
        
        # HMO for cost-conscious families
        elif family_size > 2 and income < 60000:
            return 'HMO'
        
        # Bronze plan for subsidy-eligible
        elif lead_data.get('subsidy_eligible', False):
            return 'BRONZE'
        
        else:
            return 'SILVER'

# Example usage
if __name__ == "__main__":
    scorer = HealthcareInsuranceLeadScorer()
    
    sample_healthcare_lead = {
        'lead_id': 'HC_001',
        'age': 42,
        'income': 65000,
        'family_size': 3,
        'employment_status': 'employed',
        'current_coverage': 'none',
        'health_conditions_count': 1,
        'prescription_usage': 2,
        'doctor_visits_annual': 4,
        'preventive_care_usage': 1,
        'location_risk_score': 5.5,
        'open_enrollment_period': True,
        'subsidy_eligible': False,
        'previous_claims_frequency': 1,
        'hipaa_consent_given': True,
        'consent_given': True,
        'hipaa_consent_timestamp': '2024-01-15T10:30:00Z'
    }
    
    result = scorer.score_lead(sample_healthcare_lead)
    print(f"Healthcare Lead Score: {result}")