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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceLeadScoringModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = [
            'age', 'income', 'policy_type', 'quote_requests_30d',
            'social_engagement_score', 'location_risk_score',
            'previous_insurance', 'credit_score_proxy'
        ]
        
    def preprocess_data(self, df):
        """Preprocess and engineer features"""
        # Handle categorical variables
        categorical_cols = ['policy_type', 'previous_insurance']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Feature engineering
        df['age_income_ratio'] = df['age'] / (df['income'] / 1000)
        df['engagement_per_request'] = df['social_engagement_score'] / (df['quote_requests_30d'] + 1)
        
        # Handle missing values
        df = df.fillna(df.median())
        
        return df[self.feature_columns + ['age_income_ratio', 'engagement_per_request']]
    
    def train(self, data_path):
        """Train the insurance lead scoring model"""
        logger.info("Loading training data...")
        df = pd.read_csv(data_path)
        
        # Ensure consent compliance
        df = df[df['consent_given'] == True]
        logger.info(f"Training on {len(df)} consented leads")
        
        # Preprocess features
        X = self.preprocess_data(df)
        y = df['conversion_score']  # 0-100 score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5)
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model for better performance
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Model Performance:")
        logger.info(f"MSE: {mse:.2f}")
        logger.info(f"R2 Score: {r2:.3f}")
        logger.info(f"MAE: {mae:.2f}")
        
        # Save model
        self.save_model()
        
        return {
            'mse': mse,
            'r2_score': r2,
            'mae': mae,
            'accuracy': r2  # Using R2 as accuracy proxy
        }
    
    def save_model(self):
        """Save trained model and preprocessors"""
        model_dir = 'models/insurance_lead_scoring/artifacts'
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, f'{model_dir}/model.pkl')
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{model_dir}/label_encoders.pkl')
        
        logger.info(f"Model saved to {model_dir}")

if __name__ == "__main__":
    trainer = InsuranceLeadScoringModel()
    metrics = trainer.train('data/insurance_leads_training.csv')
    print(f"Training completed with R2 score: {metrics['r2_score']:.3f}")