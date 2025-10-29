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

class LifeInsuranceLeadScoringModel(InsuranceLeadScoringModel):
    def __init__(self):
        super().__init__()
        # Life insurance-specific features
        self.feature_columns = [
            'age', 'income', 'marital_status', 'dependents_count', 'employment_status',
            'health_status', 'smoking_status', 'coverage_amount_requested', 'policy_term',
            'existing_life_insurance', 'beneficiary_count', 'debt_obligations',
            'mortgage_balance', 'education_level', 'occupation_risk_level',
            'life_stage', 'financial_dependents', 'estate_planning_needs'
        ]
        
    def preprocess_data(self, df):
        """Life insurance-specific preprocessing and feature engineering"""
        # Handle categorical variables
        categorical_cols = ['marital_status', 'employment_status', 'health_status', 
                           'smoking_status', 'education_level', 'occupation_risk_level', 'life_stage']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Life insurance-specific feature engineering
        df['age_risk_factor'] = np.where(df['age'] > 60, df['age'] * 1.5, 
                                np.where(df['age'] > 45, df['age'] * 1.2, df['age']))
        
        df['coverage_income_ratio'] = df['coverage_amount_requested'] / (df['income'] + 1)
        df['financial_responsibility_score'] = (
            df['dependents_count'] * 2 + 
            df['mortgage_balance'] / 100000 + 
            df['debt_obligations'] / 50000
        )
        
        df['mortality_risk_score'] = (
            (df['age'] / 10) + 
            (df['smoking_status'] * 3) + 
            (df['health_status'] * 2) + 
            (df['occupation_risk_level'] * 1.5)
        )
        
        df['urgency_score'] = np.where(
            (df['life_stage'] == 2) & (df['dependents_count'] > 0),  # Family stage
            8,
            np.where(df['age'] > 50, 6, 4)
        )
        
        df['affordability_score'] = np.where(
            df['coverage_income_ratio'] < 10,  # Standard rule: 10x income
            10 - df['coverage_income_ratio'],
            1
        )
        
        df['estate_planning_urgency'] = np.where(
            (df['age'] > 45) & (df['income'] > 100000),
            df['age'] / 10,
            2
        )
        
        # Handle missing values with life insurance-specific logic
        df['mortgage_balance'] = df['mortgage_balance'].fillna(0)
        df['debt_obligations'] = df['debt_obligations'].fillna(df['debt_obligations'].median())
        df['beneficiary_count'] = df['beneficiary_count'].fillna(1)
        df = df.fillna(df.median())
        
        return df[self.feature_columns + [
            'age_risk_factor', 'coverage_income_ratio', 'financial_responsibility_score',
            'mortality_risk_score', 'urgency_score', 'affordability_score', 'estate_planning_urgency'
        ]]
    
    def train(self, data_path):
        """Train life insurance lead scoring model"""
        logger.info("Loading life insurance training data...")
        df = pd.read_csv(data_path)
        
        # Ensure consent compliance
        df = df[df['consent_given'] == True]
        logger.info(f"Training on {len(df)} consented life insurance leads")
        
        # Preprocess features
        X = self.preprocess_data(df)
        y = df['conversion_score']  # 0-100 score
        
        # Stratified split considering life stage and age
        life_stage_bins = pd.cut(df['age'], bins=[0, 30, 45, 65, 100], labels=['young', 'family', 'mature', 'senior'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=life_stage_bins
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Life insurance-optimized XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.15,  # Higher regularization for life insurance
            reg_lambda=0.15,
            gamma=0.1  # Minimum split loss
        )
        
        logger.info("Training life insurance model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate with life insurance-specific metrics
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Life insurance-specific evaluation
        high_coverage_mask = X_test['coverage_amount_requested'] > 500000
        family_stage_mask = X_test['life_stage'] == 2  # Assuming 2 = family stage
        
        high_coverage_r2 = r2_score(y_test[high_coverage_mask], y_pred[high_coverage_mask]) if high_coverage_mask.sum() > 0 else 0
        family_stage_r2 = r2_score(y_test[family_stage_mask], y_pred[family_stage_mask]) if family_stage_mask.sum() > 0 else 0
        
        logger.info(f"Life Insurance Model Performance:")
        logger.info(f"Overall MSE: {mse:.2f}")
        logger.info(f"Overall R2 Score: {r2:.3f}")
        logger.info(f"Overall MAE: {mae:.2f}")
        logger.info(f"High Coverage R2: {high_coverage_r2:.3f}")
        logger.info(f"Family Stage R2: {family_stage_r2:.3f}")
        
        # Save model
        self.save_model()
        
        return {
            'mse': mse,
            'r2_score': r2,
            'mae': mae,
            'high_coverage_r2': high_coverage_r2,
            'family_stage_r2': family_stage_r2,
            'accuracy': r2
        }
    
    def save_model(self):
        """Save life insurance model"""
        model_dir = 'models/life_insurance_scoring/artifacts'
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, f'{model_dir}/model.pkl')
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{model_dir}/label_encoders.pkl')
        
        # Save feature importance for interpretability
        feature_names = self.feature_columns + [
            'age_risk_factor', 'coverage_income_ratio', 'financial_responsibility_score',
            'mortality_risk_score', 'urgency_score', 'affordability_score', 'estate_planning_urgency'
        ]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(f'{model_dir}/feature_importance.csv', index=False)
        logger.info(f"Life insurance model saved to {model_dir}")

if __name__ == "__main__":
    trainer = LifeInsuranceLeadScoringModel()
    metrics = trainer.train('data/life_insurance_leads_training.csv')
    print(f"Life insurance training completed with R2 score: {metrics['r2_score']:.3f}")
