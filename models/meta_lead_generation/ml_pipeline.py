import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import joblib
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedMLPipeline:
    """Advanced ML pipeline for lead scoring with multiple specialized models"""
    
    def __init__(self, model_path: str = None):
        self.models = {
            'conversion_probability': None,
            'lifetime_value': None,
            'churn_risk': None,
            'product_affinity': None,
            'urgency_classifier': None,
            'price_sensitivity': None,
            'engagement_predictor': None,
            'meta_ensemble': None
        }
        
        self.scalers = {
            'conversion': StandardScaler(),
            'ltv': RobustScaler(),
            'churn': StandardScaler(),
            'affinity': StandardScaler(),
            'urgency': StandardScaler(),
            'price': RobustScaler(),
            'engagement': StandardScaler(),
            'meta': StandardScaler()
        }
        
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.feature_columns = []
        
        if model_path:
            self.load_models(model_path)
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features"""
        logger.info("Creating advanced features...")
        
        # Make a copy to avoid modifying original
        features_df = df.copy()
        
        # 1. Demographic Features
        features_df = self._create_demographic_features(features_df)
        
        # 2. Financial Features
        features_df = self._create_financial_features(features_df)
        
        # 3. Behavioral Features
        features_df = self._create_behavioral_features(features_df)
        
        # 4. Temporal Features
        features_df = self._create_temporal_features(features_df)
        
        # 5. Interaction Features
        features_df = self._create_interaction_features(features_df)
        
        # 6. Risk Assessment Features
        features_df = self._create_risk_features(features_df)
        
        # 7. Market Context Features
        features_df = self._create_market_features(features_df)
        
        # 8. Social & Digital Features
        features_df = self._create_social_features(features_df)
        
        logger.info(f"Created {len(features_df.columns)} total features")
        return features_df
    
    def _create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic-based features"""
        # Age-based features
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                                labels=['young', 'young_adult', 'adult', 'middle_age', 'senior', 'elderly'])
        
        df['age_squared'] = df['age'] ** 2
        df['age_log'] = np.log1p(df['age'])
        
        # Life stage indicators
        df['is_millennial'] = ((df['age'] >= 25) & (df['age'] <= 40)).astype(int)
        df['is_gen_x'] = ((df['age'] >= 41) & (df['age'] <= 56)).astype(int)
        df['is_boomer'] = (df['age'] >= 57).astype(int)
        
        # Family composition
        if 'marital_status' in df.columns:
            df['is_married'] = (df['marital_status'] == 'married').astype(int)
            df['is_single'] = (df['marital_status'] == 'single').astype(int)
        
        if 'dependents_count' in df.columns:
            df['has_dependents'] = (df['dependents_count'] > 0).astype(int)
            df['large_family'] = (df['dependents_count'] >= 3).astype(int)
            df['dependents_per_age'] = df['dependents_count'] / (df['age'] + 1)
        
        return df
    
    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial-based features"""
        # Income features
        if 'income' in df.columns:
            df['income_log'] = np.log1p(df['income'])
            df['income_squared'] = df['income'] ** 2
            df['income_per_age'] = df['income'] / (df['age'] + 1)
            
            # Income brackets
            df['income_bracket'] = pd.cut(df['income'], 
                                        bins=[0, 30000, 50000, 75000, 100000, 150000, np.inf],
                                        labels=['low', 'lower_mid', 'mid', 'upper_mid', 'high', 'very_high'])
            
            # Affordability indicators
            df['high_income'] = (df['income'] > 100000).astype(int)
            df['low_income'] = (df['income'] < 40000).astype(int)
        
        # Credit and financial health
        if 'credit_score' in df.columns:
            df['credit_score_normalized'] = df['credit_score'] / 850.0
            df['excellent_credit'] = (df['credit_score'] >= 750).astype(int)
            df['poor_credit'] = (df['credit_score'] < 600).astype(int)
            
            if 'income' in df.columns:
                df['credit_income_ratio'] = df['credit_score'] / (df['income'] / 1000)
        
        # Debt indicators
        if 'debt_to_income' in df.columns:
            df['high_debt'] = (df['debt_to_income'] > 0.4).astype(int)
            df['low_debt'] = (df['debt_to_income'] < 0.2).astype(int)
            df['debt_risk_score'] = np.where(df['debt_to_income'] > 0.5, 1, 
                                           np.where(df['debt_to_income'] > 0.3, 0.5, 0))
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features"""
        # Engagement patterns
        if 'website_visits_30d' in df.columns:
            df['high_engagement'] = (df['website_visits_30d'] > 10).astype(int)
            df['engagement_frequency'] = df['website_visits_30d'] / 30.0
            df['engagement_log'] = np.log1p(df['website_visits_30d'])
        
        # Quote request behavior
        if 'quote_requests_30d' in df.columns:
            df['active_shopper'] = (df['quote_requests_30d'] > 3).astype(int)
            df['quote_frequency'] = df['quote_requests_30d'] / 30.0
            
            if 'website_visits_30d' in df.columns:
                df['conversion_intent'] = df['quote_requests_30d'] / (df['website_visits_30d'] + 1)
        
        # Communication preferences
        if 'preferred_contact_method' in df.columns:
            df['prefers_digital'] = df['preferred_contact_method'].isin(['email', 'text']).astype(int)
            df['prefers_phone'] = (df['preferred_contact_method'] == 'phone').astype(int)
        
        # Response patterns
        if 'response_time_hours' in df.columns:
            df['quick_responder'] = (df['response_time_hours'] <= 2).astype(int)
            df['slow_responder'] = (df['response_time_hours'] > 24).astype(int)
            df['response_urgency'] = 1 / (df['response_time_hours'] + 1)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        # Current timestamp features
        now = datetime.now()
        df['current_month'] = now.month
        df['current_quarter'] = (now.month - 1) // 3 + 1
        df['current_day_of_week'] = now.weekday()
        df['is_weekend'] = (now.weekday() >= 5).astype(int)
        
        # Seasonal features
        df['is_q4'] = (df['current_quarter'] == 4).astype(int)  # End of year planning
        df['is_spring'] = df['current_month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['current_month'].isin([6, 7, 8]).astype(int)
        
        # Lead timing
        if 'lead_created_date' in df.columns:
            df['lead_age_days'] = (now - pd.to_datetime(df['lead_created_date'])).dt.days
            df['fresh_lead'] = (df['lead_age_days'] <= 7).astype(int)
            df['stale_lead'] = (df['lead_age_days'] > 30).astype(int)
        
        # Policy renewal timing
        if 'current_policy_expiry' in df.columns:
            expiry_dates = pd.to_datetime(df['current_policy_expiry'])
            df['days_to_expiry'] = (expiry_dates - now).dt.days
            df['expiring_soon'] = (df['days_to_expiry'] <= 30).astype(int)
            df['renewal_urgency'] = np.where(df['days_to_expiry'] <= 0, 1,
                                           np.where(df['days_to_expiry'] <= 30, 0.8,
                                                  np.where(df['days_to_expiry'] <= 60, 0.5, 0.2)))
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        # Age-Income interactions
        if 'age' in df.columns and 'income' in df.columns:
            df['age_income_interaction'] = df['age'] * df['income'] / 100000
            df['income_per_life_stage'] = np.where(df['age'] < 35, df['income'] * 1.2,
                                                 np.where(df['age'] < 50, df['income'],
                                                        df['income'] * 0.8))
        
        # Family-Financial interactions
        if 'dependents_count' in df.columns and 'income' in df.columns:
            df['income_per_dependent'] = df['income'] / (df['dependents_count'] + 1)
            df['family_financial_pressure'] = df['dependents_count'] / (df['income'] / 50000)
        
        # Risk-Behavior interactions
        if 'credit_score' in df.columns and 'quote_requests_30d' in df.columns:
            df['risk_shopping_behavior'] = (df['credit_score'] / 850) * df['quote_requests_30d']
        
        # Engagement-Urgency interactions
        if 'website_visits_30d' in df.columns and 'days_to_expiry' in df.columns:
            df['urgency_engagement'] = df['website_visits_30d'] / (df['days_to_expiry'] + 1)
        
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk assessment features"""
        # Composite risk scores
        risk_factors = []
        
        # Age risk
        if 'age' in df.columns:
            df['age_risk'] = np.where(df['age'] < 25, 0.8,
                                    np.where(df['age'] < 65, 0.3, 0.6))
            risk_factors.append('age_risk')
        
        # Financial risk
        if 'credit_score' in df.columns:
            df['credit_risk'] = np.where(df['credit_score'] < 600, 0.9,
                                       np.where(df['credit_score'] < 700, 0.5, 0.1))
            risk_factors.append('credit_risk')
        
        if 'debt_to_income' in df.columns:
            df['debt_risk'] = np.where(df['debt_to_income'] > 0.5, 0.9,
                                     np.where(df['debt_to_income'] > 0.3, 0.5, 0.1))
            risk_factors.append('debt_risk')
        
        # Employment risk
        if 'employment_status' in df.columns:
            employment_risk_map = {
                'employed': 0.1,
                'self_employed': 0.4,
                'unemployed': 0.9,
                'retired': 0.3,
                'student': 0.6
            }
            df['employment_risk'] = df['employment_status'].map(employment_risk_map).fillna(0.5)
            risk_factors.append('employment_risk')
        
        # Composite risk score
        if risk_factors:
            df['composite_risk_score'] = df[risk_factors].mean(axis=1)
            df['high_risk_customer'] = (df['composite_risk_score'] > 0.6).astype(int)
            df['low_risk_customer'] = (df['composite_risk_score'] < 0.3).astype(int)
        
        return df
    
    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market context features"""
        # Location-based features
        if 'location' in df.columns:
            # High-risk states for insurance
            high_risk_states = ['FL', 'CA', 'TX', 'LA', 'NY']
            df['high_risk_location'] = df['location'].str.contains('|'.join(high_risk_states), na=False).astype(int)
            
            # Urban vs rural (simplified)
            urban_indicators = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
            df['urban_location'] = df['location'].str.contains('|'.join(urban_indicators), na=False).astype(int)
        
        # Competitive landscape features
        df['market_competitiveness'] = 0.7  # Default medium competitiveness
        df['rate_advantage'] = 0.1  # Default 10% advantage
        
        # Economic indicators (would be updated from real data)
        df['economic_confidence'] = 102.3  # Consumer confidence index
        df['unemployment_rate'] = 3.7
        df['inflation_rate'] = 3.2
        
        return df
    
    def _create_social_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create social media and digital presence features"""
        # Social media engagement
        if 'social_media_presence' in df.columns:
            df['has_social_presence'] = (df['social_media_presence'] == 'active').astype(int)
            df['limited_social_presence'] = (df['social_media_presence'] == 'limited').astype(int)
        
        # Digital behavior
        if 'online_research_behavior' in df.columns:
            df['heavy_researcher'] = (df['online_research_behavior'] == 'extensive').astype(int)
            df['quick_decider'] = (df['online_research_behavior'] == 'minimal').astype(int)
        
        # Technology adoption
        if 'preferred_contact_method' in df.columns:
            df['tech_savvy'] = df['preferred_contact_method'].isin(['email', 'app', 'chat']).astype(int)
            df['traditional_communicator'] = (df['preferred_contact_method'] == 'phone').astype(int)
        
        return df
    
    def train_conversion_probability_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train conversion probability classifier"""
        logger.info("Training conversion probability model...")
        
        # Scale features
        X_scaled = self.scalers['conversion'].fit_transform(X)
        
        # Create ensemble of classifiers
        models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        }
        
        # Train and evaluate each model
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            avg_score = cv_scores.mean()
            
            logger.info(f"{name} CV AUC: {avg_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train best model on full dataset
        best_model.fit(X_scaled, y)
        self.models['conversion_probability'] = best_model
        
        # Store feature importance
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance['conversion_probability'] = dict(zip(X.columns, best_model.feature_importances_))
        
        return {'model': best_model, 'cv_score': best_score}
    
    def train_lifetime_value_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train customer lifetime value regressor"""
        logger.info("Training lifetime value model...")
        
        # Scale features
        X_scaled = self.scalers['ltv'].fit_transform(X)
        
        # Create ensemble of regressors
        models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=300,
                depth=8,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            )
        }
        
        best_model = None
        best_score = float('-inf')
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            avg_score = cv_scores.mean()
            
            logger.info(f"{name} CV R2: {avg_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train best model
        best_model.fit(X_scaled, y)
        self.models['lifetime_value'] = best_model
        
        # Store feature importance
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance['lifetime_value'] = dict(zip(X.columns, best_model.feature_importances_))
        
        return {'model': best_model, 'cv_score': best_score}
    
    def train_urgency_classifier(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train urgency level classifier"""
        logger.info("Training urgency classifier...")
        
        # Scale features
        X_scaled = self.scalers['urgency'].fit_transform(X)
        
        # Multi-class classifier for urgency levels (LOW, MEDIUM, HIGH)
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            objective='multi:softprob'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        avg_score = cv_scores.mean()
        
        logger.info(f"Urgency classifier CV accuracy: {avg_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train on full dataset
        model.fit(X_scaled, y)
        self.models['urgency_classifier'] = model
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance['urgency_classifier'] = dict(zip(X.columns, model.feature_importances_))
        
        return {'model': model, 'cv_score': avg_score}
    
    def train_meta_ensemble(self, X: pd.DataFrame, base_predictions: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train meta-ensemble model that combines all base model predictions"""
        logger.info("Training meta-ensemble model...")
        
        # Combine original features with base model predictions
        meta_features = pd.concat([X, base_predictions], axis=1)
        
        # Scale features
        X_meta_scaled = self.scalers['meta'].fit_transform(meta_features)
        
        # Meta-learner (simple linear model for interpretability)
        meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        
        # Cross-validation
        cv_scores = cross_val_score(meta_model, X_meta_scaled, y, cv=5, scoring='r2')
        avg_score = cv_scores.mean()
        
        logger.info(f"Meta-ensemble CV R2: {avg_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train on full dataset
        meta_model.fit(X_meta_scaled, y)
        self.models['meta_ensemble'] = meta_model
        
        # Store feature importance (coefficients for linear model)
        self.feature_importance['meta_ensemble'] = dict(zip(meta_features.columns, meta_model.coef_))
        
        return {'model': meta_model, 'cv_score': avg_score}
    
    def predict_conversion_probability(self, features: np.ndarray) -> float:
        """Predict likelihood of conversion"""
        if self.models['conversion_probability']:
            scaled_features = self.scalers['conversion'].transform(features.reshape(1, -1))
            proba = self.models['conversion_probability'].predict_proba(scaled_features)[0]
            return proba[1] if len(proba) > 1 else proba[0]
        return 0.5
    
    def predict_lifetime_value(self, features: np.ndarray) -> float:
        """Predict customer lifetime value"""
        if self.models['lifetime_value']:
            scaled_features = self.scalers['ltv'].transform(features.reshape(1, -1))
            ltv = self.models['lifetime_value'].predict(scaled_features)[0]
            return max(0, ltv)
        return 10000.0
    
    def predict_urgency_level(self, features: np.ndarray) -> Tuple[str, float]:
        """Classify urgency level with confidence"""
        if self.models['urgency_classifier']:
            scaled_features = self.scalers['urgency'].transform(features.reshape(1, -1))
            proba = self.models['urgency_classifier'].predict_proba(scaled_features)[0]
            
            urgency_levels = ['LOW', 'MEDIUM', 'HIGH']
            predicted_class = np.argmax(proba)
            confidence = proba[predicted_class]
            
            return urgency_levels[predicted_class], confidence
        return "MEDIUM", 0.5
    
    def predict_comprehensive_score(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive predictions using all models"""
        # Ensure features are in correct format
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        
        feature_array = features.values
        
        # Get individual predictions
        conversion_prob = self.predict_conversion_probability(feature_array)
        lifetime_value = self.predict_lifetime_value(feature_array)
        urgency_level, urgency_confidence = self.predict_urgency_level(feature_array)
        
        # Create base predictions for meta-ensemble
        base_predictions = pd.DataFrame({
            'conversion_probability': [conversion_prob],
            'lifetime_value': [lifetime_value],
            'urgency_confidence': [urgency_confidence]
        })
        
        # Meta-ensemble prediction
        meta_score = 75.0  # Default score
        if self.models['meta_ensemble']:
            meta_features = pd.concat([features.reset_index(drop=True), base_predictions], axis=1)
            meta_scaled = self.scalers['meta'].transform(meta_features)
            meta_score = self.models['meta_ensemble'].predict(meta_scaled)[0]
            meta_score = max(0, min(100, meta_score))  # Clamp to 0-100
        
        return {
            'overall_score': meta_score,
            'conversion_probability': conversion_prob,
            'lifetime_value': lifetime_value,
            'urgency_level': urgency_level,
            'urgency_confidence': urgency_confidence,
            'model_confidence': self._calculate_prediction_confidence(features),
            'score_components': {
                'conversion_weight': 0.4,
                'ltv_weight': 0.3,
                'urgency_weight': 0.2,
                'confidence_weight': 0.1
            }
        }
    
    def _calculate_prediction_confidence(self, features: pd.DataFrame) -> float:
        """Calculate overall prediction confidence"""
        # Simple confidence based on feature completeness and model agreement
        feature_completeness = features.notna().sum().sum() / features.size
        
        # Model agreement (simplified - would compare predictions across models)
        model_agreement = 0.8  # Placeholder
        
        return (feature_completeness * 0.6 + model_agreement * 0.4)
    
    def get_feature_importance(self, model_type: str = 'all') -> Dict[str, Any]:
        """Get feature importance for interpretability"""
        if model_type == 'all':
            return self.feature_importance
        else:
            return self.feature_importance.get(model_type, {})
    
    def save_models(self, model_path: str):
        """Save all trained models and preprocessors"""
        import os
        os.makedirs(model_path, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            if model is not None:
                joblib.dump(model, f'{model_path}/{name}_model.pkl')
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{model_path}/{name}_scaler.pkl')
        
        # Save other components
        joblib.dump(self.label_encoders, f'{model_path}/label_encoders.pkl')
        joblib.dump(self.feature_importance, f'{model_path}/feature_importance.pkl')
        joblib.dump(self.feature_columns, f'{model_path}/feature_columns.pkl')
        
        logger.info(f"Models saved to {model_path}")
    
    def load_models(self, model_path: str):
        """Load all trained models and preprocessors"""
        import os
        
        try:
            # Load models
            for name in self.models.keys():
                model_file = f'{model_path}/{name}_model.pkl'
                if os.path.exists(model_file):
                    self.models[name] = joblib.load(model_file)
            
            # Load scalers
            for name in self.scalers.keys():
                scaler_file = f'{model_path}/{name}_scaler.pkl'
                if os.path.exists(scaler_file):
                    self.scalers[name] = joblib.load(scaler_file)
            
            # Load other components
            if os.path.exists(f'{model_path}/label_encoders.pkl'):
                self.label_encoders = joblib.load(f'{model_path}/label_encoders.pkl')
            
            if os.path.exists(f'{model_path}/feature_importance.pkl'):
                self.feature_importance = joblib.load(f'{model_path}/feature_importance.pkl')
            
            if os.path.exists(f'{model_path}/feature_columns.pkl'):
                self.feature_columns = joblib.load(f'{model_path}/feature_columns.pkl')
            
            logger.info(f"Models loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# Usage example and training pipeline
class MLPipelineTrainer:
    """Training pipeline for the advanced ML models"""
    
    def __init__(self):
        self.pipeline = AdvancedMLPipeline()
    
    def train_full_pipeline(self, training_data_path: str) -> Dict[str, Any]:
        """Train the complete ML pipeline"""
        logger.info("Starting full pipeline training...")
        
        # Load training data
        df = pd.read_csv(training_data_path)
        
        # Create advanced features
        features_df = self.pipeline.create_advanced_features(df)
        
        # Prepare target variables (assuming they exist in the data)
        targets = {
            'conversion': df.get('converted', pd.Series([0] * len(df))),
            'lifetime_value': df.get('customer_lifetime_value', pd.Series([10000] * len(df))),
            'urgency': df.get('urgency_level', pd.Series(['MEDIUM'] * len(df)))
        }
        
        # Store feature columns
        self.pipeline.feature_columns = features_df.columns.tolist()
        
        # Train individual models
        results = {}
        
        # 1. Conversion probability model
        if 'converted' in df.columns:
            conv_result = self.pipeline.train_conversion_probability_model(
                features_df, targets['conversion']
            )
            results['conversion_model'] = conv_result
        
        # 2. Lifetime value model
        if 'customer_lifetime_value' in df.columns:
            ltv_result = self.pipeline.train_lifetime_value_model(
                features_df, targets['lifetime_value']
            )
            results['ltv_model'] = ltv_result
        
        # 3. Urgency classifier
        if 'urgency_level' in df.columns:
            urgency_result = self.pipeline.train_urgency_classifier(
                features_df, targets['urgency']
            )
            results['urgency_model'] = urgency_result
        
        # 4. Meta-ensemble (if we have a composite target)
        if 'overall_score' in df.columns:
            # Create base predictions for meta-learning
            base_preds = pd.DataFrame({
                'conversion_prob': [self.pipeline.predict_conversion_probability(row.values) 
                                  for _, row in features_df.iterrows()],
                'ltv_pred': [self.pipeline.predict_lifetime_value(row.values) 
                           for _, row in features_df.iterrows()]
            })
            
            meta_result = self.pipeline.train_meta_ensemble(
                features_df, base_preds, df['overall_score']
            )
            results['meta_ensemble'] = meta_result
        
        logger.info("Pipeline training completed!")
        return results

if __name__ == "__main__":
    # Example usage
    trainer = MLPipelineTrainer()
    
    # Train the pipeline (assuming training data exists)
    # results = trainer.train_full_pipeline('data/training_data.csv')
    
    # Save the trained pipeline
    # trainer.pipeline.save_models('models/meta_lead_generation/ml_pipeline')
    
    print("Advanced ML Pipeline ready for training!")