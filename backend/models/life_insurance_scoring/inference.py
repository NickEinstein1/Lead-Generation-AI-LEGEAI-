import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Union
import hashlib
from datetime import datetime, date, timezone
import sys
sys.path.append('../../')
from backend.models.insurance_lead_scoring.inference import InsuranceLeadScorer
from backend.models.insurance_products import (
    LifeInsurancePolicyType,
    get_recommended_policy_types,
    get_policy_display_name,
    LIFE_INSURANCE_PRODUCTS
)

logger = logging.getLogger(__name__)

class LifeInsuranceLeadScorer(InsuranceLeadScorer):
    def __init__(self, model_path='models/life_insurance_scoring/artifacts'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.load_model()

    def load_model(self):
        """Load trained model and preprocessors"""
        try:
            self.model = joblib.load(f'{self.model_path}/model.pkl')
            self.scaler = joblib.load(f'{self.model_path}/scaler.pkl')
            self.label_encoders = joblib.load(f'{self.model_path}/label_encoders.pkl')
            logger.info("Life insurance model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading life insurance model: {e}")
            raise

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
        
        df['coverage_income_ratio'] = df['coverage_amount_requested'] / (df['income'] + 1)
        df['financial_responsibility_score'] = (
            df['dependents_count'] * 2 +
            df.get('mortgage_balance', pd.Series([0])).fillna(0) / 100000 +
            df.get('debt_obligations', pd.Series([0])).fillna(0) / 50000
        )
        
        df['mortality_risk_score'] = self.calculate_mortality_risk(lead_data)
        
        life_stage_map = {'young_professional': 0, 'family_building': 1, 'wealth_accumulation': 2, 'estate_planning': 3}
        life_stage = self.determine_life_stage(lead_data)
        df['life_stage'] = life_stage_map.get(life_stage, 1)
        
        df['urgency_score'] = np.where(
            (df['life_stage'] == 1) & (df['dependents_count'] > 0),  # Family building
            8,
            np.where(df['age'] > 50, 6, 4)
        )

        df['affordability_score'] = np.where(
            df['coverage_income_ratio'] < 10,
            10 - df['coverage_income_ratio'],
            1
        )

        df['estate_planning_urgency'] = np.where(
            (df['age'] > 45) & (df['income'] > 100000),
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
        if 'mortgage_balance' in df.columns:
            df['mortgage_balance'] = df['mortgage_balance'].fillna(0)
        if 'debt_obligations' in df.columns:
            df['debt_obligations'] = df['debt_obligations'].fillna(0)
        if 'beneficiary_count' in df.columns:
            df['beneficiary_count'] = df['beneficiary_count'].fillna(1)
        if 'existing_life_insurance' in df.columns:
            df['existing_life_insurance'] = df['existing_life_insurance'].fillna(0)
        if 'financial_dependents' in df.columns:
            df['financial_dependents'] = df['financial_dependents'].fillna(df['dependents_count'])
        if 'estate_planning_needs' in df.columns:
            df['estate_planning_needs'] = df['estate_planning_needs'].fillna(0)

        # Fill any remaining missing values
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
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
                    'timestamp': datetime.now(timezone.utc).isoformat(),
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

            # Get detailed policy recommendations
            policy_recommendations = self._get_policy_recommendations(lead_data)

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
                'policy_recommendations': policy_recommendations,
                'urgency_level': self._get_urgency_level(lead_data),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'model_version': '1.0_life_insurance',
                'compliance_status': 'PASSED'
            }
            
        except Exception as e:
            logger.error(f"Error scoring life insurance lead {lead_data.get('lead_id')}: {e}")
            return {
                'lead_id': lead_data.get('lead_id'),
                'score': 0,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat(),
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
    
    def _calculate_confidence(self, X: np.ndarray) -> float:
        """Calculate prediction confidence"""
        try:
            # For XGBoost, we can use prediction with output_margin
            margin = self.model.predict(X, output_margin=True)[0]
            confidence = 1 / (1 + np.exp(-abs(margin)))
            return round(float(confidence), 3)
        except:
            return 0.65  # Default confidence

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
        """
        Recommend life insurance policy type based on comprehensive underwriting criteria
        Returns the policy type enum value as string
        """
        age = lead_data.get('age', 30)
        income = lead_data.get('income', 50000)
        life_stage = self.determine_life_stage(lead_data)
        coverage_amount = lead_data.get('coverage_amount_requested', 0)
        dependents = lead_data.get('dependents_count', 0)
        goal = lead_data.get('primary_goal', 'income_replacement')

        # Determine primary goal if not specified
        if not goal or goal == 'income_replacement':
            if life_stage == 'estate_planning' and income > 150000:
                goal = 'estate_planning'
            elif age >= 50 and coverage_amount < 50000:
                goal = 'final_expense'
            elif age >= 45 and life_stage in ['wealth_accumulation', 'estate_planning']:
                goal = 'retirement_income'
            elif income > 100000 and age > 35:
                goal = 'wealth_accumulation'
            else:
                goal = 'income_replacement'

        # Get recommendations based on profile
        recommendations = get_recommended_policy_types(age, income, goal)

        # Additional logic for specific scenarios
        if not recommendations:
            # Fallback logic
            if age < 40 and dependents > 0:
                return LifeInsurancePolicyType.TERM_LIFE.value
            elif income > 200000 and age < 60:
                return LifeInsurancePolicyType.INDEXED_UNIVERSAL_LIFE.value
            elif age >= 50 and coverage_amount < 50000:
                return LifeInsurancePolicyType.FINAL_EXPENSE.value
            elif age >= 45:
                return LifeInsurancePolicyType.WHOLE_LIFE.value
            else:
                return LifeInsurancePolicyType.TERM_LIFE.value

        # Return the first (best) recommendation
        return recommendations[0].value

    def _get_policy_recommendations(self, lead_data: Dict) -> List[Dict]:
        """
        Get multiple policy type recommendations with details
        Returns a list of recommended policies with full information
        """
        age = lead_data.get('age', 30)
        income = lead_data.get('income', 50000)
        life_stage = self.determine_life_stage(lead_data)
        coverage_amount = lead_data.get('coverage_amount_requested', 0)
        goal = lead_data.get('primary_goal', 'income_replacement')

        # Determine goal
        if not goal or goal == 'income_replacement':
            if life_stage == 'estate_planning' and income > 150000:
                goal = 'estate_planning'
            elif age >= 50 and coverage_amount < 50000:
                goal = 'final_expense'
            elif age >= 45:
                goal = 'retirement_income'
            elif income > 100000:
                goal = 'wealth_accumulation'
            else:
                goal = 'income_replacement'

        # Get recommendations
        policy_types = get_recommended_policy_types(age, income, goal)

        # Build detailed recommendations
        recommendations = []
        for policy_type in policy_types[:3]:  # Top 3 recommendations
            if policy_type in LIFE_INSURANCE_PRODUCTS:
                product_info = LIFE_INSURANCE_PRODUCTS[policy_type]

                # Calculate estimated premium (simplified)
                base_rate = 0.005 if product_info.category == "term" else 0.012
                if product_info.investment_component:
                    base_rate *= 1.5

                estimated_premium = coverage_amount * base_rate * (age / 40)

                recommendations.append({
                    'policy_type': policy_type.value,
                    'display_name': product_info.display_name,
                    'category': product_info.category,
                    'description': product_info.description,
                    'estimated_monthly_premium': round(estimated_premium / 12, 2),
                    'estimated_annual_premium': round(estimated_premium, 2),
                    'cash_value': product_info.cash_value,
                    'investment_component': product_info.investment_component,
                    'best_for': product_info.best_for,
                    'key_features': product_info.key_features,
                    'underwriting_complexity': product_info.underwriting_complexity
                })

        return recommendations

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
