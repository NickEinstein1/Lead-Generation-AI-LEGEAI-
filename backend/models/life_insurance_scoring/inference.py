import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Union
import hashlib
from datetime import datetime, date
import sys
sys.path.append('../../')
from backend.models.insurance_lead_scoring.inference import InsuranceLeadScorer

logger = logging.getLogger(__name__)

class LifeInsuranceLeadScorer(InsuranceLeadScorer):
    def __init__(self, model_path='models/life_insurance_scoring/artifacts'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.load_model()
        
    def calculate_mortality_risk(self, lead_data: Dict) -> float:
        """Calculate mortality risk score for life insurance"""
        age = lead_data.get('age', 30)
        smoking_status = 1 if lead_data.get('smoking_status') == 'smoker' else 0
        health_status = lead_data.get('health_status', 'good')
        occupation_risk = lead_data.get('occupation_risk_level', 'low')
        
        # Health status mapping
        health_score = {'excellent': 0, 'good': 1, 'fair': 2, 'poor': 3}.get(health_status, 1)
        
        # Occupation risk mapping
        occupation_score = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}.get(occupation_risk, 0)
        
        mortality_risk = (age / 10) + (smoking_status * 3) + (health_score * 2) + (occupation_score * 1.5)
        
        return min(mortality_risk, 20)  # Cap at 20
    
    def calculate_coverage_adequacy(self, lead_data: Dict) -> Dict:
        """Calculate coverage adequacy and recommendations"""
        income = lead_data.get('income', 50000)
        dependents = lead_data.get('dependents_count', 0)
        mortgage = lead_data.get('mortgage_balance', 0)
        debt = lead_data.get('debt_obligations', 0)
        requested_coverage = lead_data.get('coverage_amount_requested', 0)
        
        # Standard coverage calculation: 10x income + debts + education costs
        education_cost = dependents * 50000  # Estimated per child
        recommended_coverage = (income * 10) + mortgage + debt + education_cost
        
        adequacy_ratio = requested_coverage / recommended_coverage if recommended_coverage > 0 else 0
        
        return {
            'recommended_coverage': recommended_coverage,
            'adequacy_ratio': adequacy_ratio,
            'coverage_gap': max(0, recommended_coverage - requested_coverage),
            'adequacy_level': 'adequate' if adequacy_ratio >= 0.8 else 'insufficient'
        }
    
    def determine_life_stage(self, lead_data: Dict) -> str:
        """Determine life stage for targeted recommendations"""
        age = lead_data.get('age', 30)
        marital_status = lead_data.get('marital_status', 'single')
        dependents = lead_data.get('dependents_count', 0)
        
        if age < 30:
            return 'young_professional'
        elif age < 45 and (marital_status in ['married', 'partnered'] or dependents > 0):
            return 'family_building'
        elif age < 60:
            return 'wealth_accumulation'
        else:
            return 'estate_planning'
    
    def preprocess_lead(self, lead_data: Dict) -> np.ndarray:
        """Life insurance-specific lead preprocessing"""
        # Convert to DataFrame
        df = pd.DataFrame([lead_data])
        
        # Handle categorical variables
        categorical_cols = ['marital_status', 'employment_status', 'health_status', 
                           'smoking_status', 'education_level', 'occupation_risk_level', 'life_stage']
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Life insurance feature engineering
        df['age_risk_factor'] = np.where(df['age'] > 60, df['age'] * 1.5, 
                                np.where(df['age'] > 45, df['age'] * 1.2, df['age']))
        
        df['coverage_income_ratio'] = df.get('coverage_amount_requested', 0) / (df.get('income', 1) + 1)
        df['financial_responsibility_score'] = (
            df.get('dependents_count', 0) * 2 + 
            df.get('mortgage_balance', 0) / 100000 + 
            df.get('debt_obligations', 0) / 50000
        )
        
        df['mortality_risk_score'] = self.calculate_mortality_risk(lead_data)
        
        life_stage_map = {'young_professional': 0, 'family_building': 1, 'wealth_accumulation': 2, 'estate_planning': 3}
        life_stage = self.determine_life_stage(lead_data)
        df['life_stage'] = life_stage_map.get(life_stage, 1)
        
        df['urgency_score'] = np.where(
            (df['life_stage'] == 1) & (df.get('dependents_count', 0) > 0),  # Family building
            8,
            np.where(df['age'] > 50, 6, 4)
        )
        
        df['affordability_score'] = np.where(
            df['coverage_income_ratio'] < 10,
            10 - df['coverage_income_ratio'],
            1
        )
        
        df['estate_planning_urgency'] = np.where(
            (df['age'] > 45) & (df.get('income', 0) > 100000),
            df['age'] / 10,
            2
        )
        
        # Feature columns
        feature_columns = [
            'age', 'income', 'marital_status', 'dependents_count', 'employment_status',
            'health_status', 'smoking_status', 'coverage_amount_requested', 'policy_term',
            'existing_life_insurance', 'beneficiary_count', 'debt_obligations',
            'mortgage_balance', 'education_level', 'occupation_risk_level',
            'life_stage', 'financial_dependents', 'estate_planning_needs',
            'age_risk_factor', 'coverage_income_ratio', 'financial_responsibility_score',
            'mortality_risk_score', 'urgency_score', 'affordability_score', 'estate_planning_urgency'
        ]
        
        # Handle missing values with life insurance defaults
        df['mortgage_balance'] = df['mortgage_balance'].fillna(0)
        df['debt_obligations'] = df['debt_obligations'].fillna(0)
        df['beneficiary_count'] = df['beneficiary_count'].fillna(1)
        df['existing_life_insurance'] = df['existing_life_insurance'].fillna(0)
        df['financial_dependents'] = df['financial_dependents'].fillna(df.get('dependents_count', 0))
        df['estate_planning_needs'] = df['estate_planning_needs'].fillna(0)
        df = df.fillna(df.median())
        
        # Scale features
        X = self.scaler.transform(df[feature_columns])
        
        return X
    
    def score_lead(self, lead_data: Dict) -> Dict:
        """Score life insurance lead with comprehensive analysis"""
        try:
            # Validate consent
            if not self.validate_consent(lead_data):
                return {
                    'lead_id': lead_data.get('lead_id'),
                    'score': 0,
                    'error': 'Consent not provided or invalid',
                    'timestamp': datetime.utcnow().isoformat(),
                    'compliance_status': 'FAILED'
                }
            
            # Anonymize PII
            anonymized_data = self.anonymize_pii(lead_data)
            
            # Preprocess
            X = self.preprocess_lead(anonymized_data)
            
            # Predict base score
            base_score = self.model.predict(X)[0]
            
            # Life insurance-specific adjustments
            adjusted_score = self._apply_life_insurance_adjustments(base_score, lead_data)
            
            # Coverage analysis
            coverage_analysis = self.calculate_coverage_adequacy(lead_data)
            
            # Ensure score is within 0-100 range
            final_score = max(0, min(100, adjusted_score))
            
            return {
                'lead_id': lead_data.get('lead_id'),
                'score': round(float(final_score), 2),
                'base_score': round(float(base_score), 2),
                'confidence': self._calculate_confidence(X),
                'life_stage': self.determine_life_stage(lead_data),
                'mortality_risk_score': self.calculate_mortality_risk(lead_data),
                'recommended_coverage': coverage_analysis['recommended_coverage'],
                'coverage_adequacy': coverage_analysis['adequacy_level'],
                'coverage_gap': coverage_analysis['coverage_gap'],
                'recommended_policy_type': self._recommend_policy_type(lead_data),
                'urgency_level': self._get_urgency_level(lead_data),
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': '1.0_life_insurance',
                'compliance_status': 'PASSED'
            }
            
        except Exception as e:
            logger.error(f"Error scoring life insurance lead {lead_data.get('lead_id')}: {e}")
            return {
                'lead_id': lead_data.get('lead_id'),
                'score': 0,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'compliance_status': 'ERROR'
            }
    
    def _apply_life_insurance_adjustments(self, base_score: float, lead_data: Dict) -> float:
        """Apply life insurance-specific score adjustments"""
        adjusted_score = base_score
        
        # Family stage boost
        if self.determine_life_stage(lead_data) == 'family_building':
            adjusted_score *= 1.2
        
        # High coverage amount boost
        if lead_data.get('coverage_amount_requested', 0) > 500000:
            adjusted_score *= 1.1
        
        # Health status adjustment
        health_status = lead_data.get('health_status', 'good')
        if health_status == 'excellent':
            adjusted_score *= 1.05
        elif health_status in ['fair', 'poor']:
            adjusted_score *= 0.9
        
        # Smoking penalty
        if lead_data.get('smoking_status') == 'smoker':
            adjusted_score *= 0.85
        
        # No existing coverage boost
        if not lead_data.get('existing_life_insurance', False):
            adjusted_score *= 1.08
        
        return adjusted_score
    
    def _get_urgency_level(self, lead_data: Dict) -> str:
        """Determine urgency level for life insurance lead"""
        life_stage = self.determine_life_stage(lead_data)
        age = lead_data.get('age', 30)
        dependents = lead_data.get('dependents_count', 0)
        
        if life_stage == 'family_building' and dependents > 0:
            return 'CRITICAL'
        elif age > 55 or dependents > 2:
            return 'HIGH'
        elif life_stage in ['wealth_accumulation', 'estate_planning']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _recommend_policy_type(self, lead_data: Dict) -> str:
        """Recommend life insurance policy type"""
        age = lead_data.get('age', 30)
        income = lead_data.get('income', 50000)
        life_stage = self.determine_life_stage(lead_data)
        coverage_amount = lead_data.get('coverage_amount_requested', 0)
        
        # Term life for young families and temporary needs
        if life_stage in ['young_professional', 'family_building'] and age < 45:
            return 'TERM_LIFE'
        
        # Whole life for estate planning and permanent needs
        elif life_stage == 'estate_planning' or income > 150000:
            return 'WHOLE_LIFE'
        
        # Universal life for flexible premium needs
        elif income > 100000 and age > 40:
            return 'UNIVERSAL_LIFE'
        
        # Variable life for investment-minded clients
        elif income > 200000 and age < 55:
            return 'VARIABLE_LIFE'
        
        else:
            return 'TERM_LIFE'  # Default recommendation

# Example usage
if __name__ == "__main__":
    scorer = LifeInsuranceLeadScorer()
    
    sample_life_lead = {
        'lead_id': 'LIFE_001',
        'age': 35,
        'income': 85000,
        'marital_status': 'married',
        'dependents_count': 2,
        'employment_status': 'employed',
        'health_status': 'good',
        'smoking_status': 'non_smoker',
        'coverage_amount_requested': 750000,
        'policy_term': 20,
        'existing_life_insurance': False,
        'beneficiary_count': 2,
        'debt_obligations': 25000,
        'mortgage_balance': 300000,
        'education_level': 'bachelors',
        'occupation_risk_level': 'low',
        'financial_dependents': 2,
        'estate_planning_needs': 1,
        'consent_given': True,
        'consent_timestamp': '2024-01-15T10:30:00Z'
    }
    
    result = scorer.score_lead(sample_life_lead)
    print(f"Life Insurance Lead Score: {result}")
