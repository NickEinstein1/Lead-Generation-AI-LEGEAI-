import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import logging
from datetime import datetime
import os
import sys
sys.path.append('../../')
from backend.models.insurance_lead_scoring.train import InsuranceLeadScoringModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareInsuranceLeadScoringModel(InsuranceLeadScoringModel):
    def __init__(self):
        super().__init__()
        # Healthcare-specific features
        self.feature_columns = [
            'age', 'income', 'family_size', 'employment_status',
            'current_coverage', 'health_conditions_count', 'prescription_usage',
            'doctor_visits_annual', 'preventive_care_usage', 'location_risk_score',
            'open_enrollment_period', 'subsidy_eligible', 'previous_claims_frequency'
        ]
        
    def preprocess_data(self, df):
        """Healthcare-specific preprocessing and feature engineering"""
        # Handle categorical variables
        categorical_cols = ['employment_status', 'current_coverage']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Healthcare-specific feature engineering
        df['age_risk_factor'] = np.where(df['age'] > 50, df['age'] * 1.2, df['age'])
        df['health_complexity_score'] = (
            df['health_conditions_count'] * 2 + 
            df['prescription_usage'] + 
            df['doctor_visits_annual'] / 12
        )
        df['affordability_ratio'] = df['income'] / (df['family_size'] * 12000)  # Rough healthcare cost estimate
        df['urgency_score'] = np.where(
            (df['current_coverage'] == 0) & (df['health_conditions_count'] > 0), 
            10, 
            df['open_enrollment_period'] * 3
        )
        df['subsidy_need_score'] = np.where(
            df['subsidy_eligible'] == 1,
            (400 - (df['income'] / 100)) / 100,  # Higher score for lower income
            0
        )
        
        # Handle missing values with healthcare-specific logic
        df['prescription_usage'] = df['prescription_usage'].fillna(df['prescription_usage'].median())
        df['doctor_visits_annual'] = df['doctor_visits_annual'].fillna(2)  # Average visits
        df = df.fillna(df.median())
        
        return df[self.feature_columns + [
            'age_risk_factor', 'health_complexity_score', 'affordability_ratio',
            'urgency_score', 'subsidy_need_score'
        ]]
    
    def train(self, data_path):
        """Train healthcare insurance lead scoring model"""
        logger.info("Loading healthcare insurance training data...")
        df = pd.read_csv(data_path)
        
        # HIPAA compliance - ensure consent and anonymization
        df = df[df['hipaa_consent_given'] == True]
        logger.info(f"Training on {len(df)} HIPAA-consented healthcare leads")
        
        # Preprocess features
        X = self.preprocess_data(df)
        y = df['conversion_score']  # 0-100 score
        
        # Stratified split considering healthcare urgency
        urgency_bins = pd.cut(df['urgency_score'], bins=3, labels=['low', 'medium', 'high'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=urgency_bins
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Healthcare-optimized XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,  # L1 regularization for feature selection
            reg_lambda=0.1   # L2 regularization
        )
        
        logger.info("Training healthcare insurance model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate with healthcare-specific metrics
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Healthcare-specific evaluation
        high_urgency_mask = X_test['urgency_score'] > 7
        if high_urgency_mask.sum() > 0:
            high_urgency_r2 = r2_score(
                y_test[high_urgency_mask], 
                y_pred[high_urgency_mask]
            )
        else:
            high_urgency_r2 = 0
        
        logger.info(f"Healthcare Model Performance:")
        logger.info(f"Overall MSE: {mse:.2f}")
        logger.info(f"Overall R2 Score: {r2:.3f}")
        logger.info(f"Overall MAE: {mae:.2f}")
        logger.info(f"High Urgency R2: {high_urgency_r2:.3f}")
        
        # Save model
        self.save_model()
        
        return {
            'mse': mse,
            'r2_score': r2,
            'mae': mae,
            'high_urgency_r2': high_urgency_r2,
            'accuracy': r2
        }
    
    def save_model(self):
        """Save healthcare insurance model"""
        model_dir = 'models/healthcare_insurance_scoring/artifacts'
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, f'{model_dir}/model.pkl')
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{model_dir}/label_encoders.pkl')
        
        # Save feature importance for interpretability
        feature_names = self.feature_columns + [
            'age_risk_factor', 'health_complexity_score', 'affordability_ratio',
            'urgency_score', 'subsidy_need_score'
        ]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(f'{model_dir}/feature_importance.csv', index=False)
        logger.info(f"Healthcare model saved to {model_dir}")

if __name__ == "__main__":
    trainer = HealthcareInsuranceLeadScoringModel()
    metrics = trainer.train('data/healthcare_insurance_leads_training.csv')
    print(f"Healthcare training completed with R2 score: {metrics['r2_score']:.3f}")
