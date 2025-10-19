"""
Behavioral Prediction Engine

Advanced behavioral analytics for predicting next best actions,
churn risk, engagement patterns, and customer lifecycle events.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

class NextBestAction(Enum):
    EMAIL_FOLLOW_UP = "email_follow_up"
    PHONE_CALL = "phone_call"
    SEND_QUOTE = "send_quote"
    SCHEDULE_MEETING = "schedule_meeting"
    SEND_CONTENT = "send_content"
    NURTURE_SEQUENCE = "nurture_sequence"
    DIRECT_OUTREACH = "direct_outreach"
    WAIT_AND_MONITOR = "wait_and_monitor"

class ChurnRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EngagementPattern(Enum):
    HIGHLY_ENGAGED = "highly_engaged"
    MODERATELY_ENGAGED = "moderately_engaged"
    DECLINING_ENGAGEMENT = "declining_engagement"
    DORMANT = "dormant"
    RE_ENGAGING = "re_engaging"

@dataclass
class BehavioralInsight:
    """Behavioral prediction result"""
    lead_id: str
    next_best_action: NextBestAction
    action_confidence: float
    churn_risk: ChurnRiskLevel
    churn_probability: float
    engagement_pattern: EngagementPattern
    engagement_score: float
    predicted_conversion_timeline: int  # days
    recommended_touchpoints: List[str]
    behavioral_triggers: List[str]
    risk_factors: List[str]
    opportunity_indicators: List[str]
    prediction_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BehavioralFeatures:
    """Behavioral feature set for prediction"""
    # Engagement metrics
    email_open_rate: float = 0.0
    email_click_rate: float = 0.0
    website_visit_frequency: float = 0.0
    page_views_per_session: float = 0.0
    time_on_site: float = 0.0
    
    # Interaction patterns
    days_since_last_interaction: int = 0
    total_interactions: int = 0
    interaction_frequency: float = 0.0
    response_rate: float = 0.0
    
    # Content engagement
    content_downloads: int = 0
    webinar_attendance: int = 0
    social_media_engagement: float = 0.0
    
    # Communication preferences
    preferred_channel: str = "email"
    best_contact_time: str = "morning"
    communication_frequency_preference: str = "weekly"
    
    # Lifecycle stage
    days_in_pipeline: int = 0
    stage_progression_rate: float = 0.0
    stagnation_periods: int = 0
    
    # Behavioral changes
    engagement_trend: str = "stable"  # increasing, decreasing, stable
    activity_pattern_change: float = 0.0
    seasonal_behavior_factor: float = 1.0

class BehavioralPredictionEngine:
    """Advanced behavioral prediction and analytics engine"""
    
    def __init__(self):
        # Prediction models
        self.next_action_model = None
        self.churn_risk_model = None
        self.engagement_model = None
        self.timeline_model = None
        
        # Feature processors
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        
        # Model performance
        self.model_performance = {}
        
        # Behavioral patterns
        self.behavioral_patterns = {}
        self.engagement_thresholds = {
            'highly_engaged': 0.8,
            'moderately_engaged': 0.6,
            'declining_engagement': 0.4,
            'dormant': 0.2
        }
        
        logger.info("Behavioral Prediction Engine initialized")
    
    async def train_behavioral_models(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """Train all behavioral prediction models"""
        
        try:
            logger.info("Training behavioral prediction models...")
            
            # Prepare features
            features = self._extract_behavioral_features(training_data)
            
            # Train individual models
            results = {}
            
            # Next best action model
            if 'next_action' in training_data.columns:
                results['next_action'] = await self._train_next_action_model(features, training_data['next_action'])
            
            # Churn risk model
            if 'churned' in training_data.columns:
                results['churn_risk'] = await self._train_churn_risk_model(features, training_data['churned'])
            
            # Engagement model
            if 'engagement_score' in training_data.columns:
                results['engagement'] = await self._train_engagement_model(features, training_data['engagement_score'])
            
            # Timeline model
            if 'conversion_timeline' in training_data.columns:
                results['timeline'] = await self._train_timeline_model(features, training_data['conversion_timeline'])
            
            logger.info("Behavioral models training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error training behavioral models: {e}")
            raise
    
    async def _train_next_action_model(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Train next best action prediction model"""
        
        # Encode target labels
        le = LabelEncoder()
        targets_encoded = le.fit_transform(targets)
        self.label_encoders['next_action'] = le
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Train ensemble model
        models = {
            'xgb': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'lgb': lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            scores = cross_val_score(model, features_scaled, targets_encoded, cv=5, scoring='accuracy')
            avg_score = scores.mean()
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train best model on full dataset
        best_model.fit(features_scaled, targets_encoded)
        self.next_action_model = best_model
        
        self.model_performance['next_action'] = best_score
        return best_score
    
    async def _train_churn_risk_model(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Train churn risk prediction model"""
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Train gradient boosting model
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Cross-validation
        scores = cross_val_score(model, features_scaled, targets, cv=5, scoring='roc_auc')
        avg_score = scores.mean()
        
        # Train on full dataset
        model.fit(features_scaled, targets)
        self.churn_risk_model = model
        
        self.model_performance['churn_risk'] = avg_score
        return avg_score
    
    async def _train_engagement_model(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Train engagement score prediction model"""
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Train regression model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        
        # Cross-validation
        scores = cross_val_score(model, features_scaled, targets, cv=5, scoring='r2')
        avg_score = scores.mean()
        
        # Train on full dataset
        model.fit(features_scaled, targets)
        self.engagement_model = model
        
        self.model_performance['engagement'] = avg_score
        return avg_score
    
    async def _train_timeline_model(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Train conversion timeline prediction model"""
        
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Train regression model
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Cross-validation
        scores = cross_val_score(model, features_scaled, targets, cv=5, scoring='r2')
        avg_score = scores.mean()
        
        # Train on full dataset
        model.fit(features_scaled, targets)
        self.timeline_model = model
        
        self.model_performance['timeline'] = avg_score
        return avg_score
    
    async def predict_behavioral_insights(self, lead_data: Dict[str, Any]) -> BehavioralInsight:
        """Generate comprehensive behavioral predictions for a lead"""
        
        try:
            # Extract behavioral features
            features = self._extract_single_lead_features(lead_data)
            features_scaled = self.feature_scaler.transform([features])
            
            # Predict next best action
            next_action = NextBestAction.EMAIL_FOLLOW_UP
            action_confidence = 0.5
            
            if self.next_action_model:
                action_proba = self.next_action_model.predict_proba(features_scaled)[0]
                action_idx = np.argmax(action_proba)
                action_confidence = action_proba[action_idx]
                
                action_label = self.label_encoders['next_action'].inverse_transform([action_idx])[0]
                next_action = NextBestAction(action_label)
            
            # Predict churn risk
            churn_probability = 0.3
            churn_risk = ChurnRiskLevel.MEDIUM
            
            if self.churn_risk_model:
                churn_proba = self.churn_risk_model.predict_proba(features_scaled)[0]
                churn_probability = churn_proba[1] if len(churn_proba) > 1 else churn_proba[0]
                
                if churn_probability > 0.8:
                    churn_risk = ChurnRiskLevel.CRITICAL
                elif churn_probability > 0.6:
                    churn_risk = ChurnRiskLevel.HIGH
                elif churn_probability > 0.4:
                    churn_risk = ChurnRiskLevel.MEDIUM
                else:
                    churn_risk = ChurnRiskLevel.LOW
            
            # Predict engagement
            engagement_score = 0.6
            engagement_pattern = EngagementPattern.MODERATELY_ENGAGED
            
            if self.engagement_model:
                engagement_score = self.engagement_model.predict(features_scaled)[0]
                engagement_pattern = self._determine_engagement_pattern(engagement_score, lead_data)
            
            # Predict timeline
            predicted_timeline = 30
            if self.timeline_model:
                predicted_timeline = max(1, int(self.timeline_model.predict(features_scaled)[0]))
            
            # Generate recommendations
            touchpoints = self._recommend_touchpoints(next_action, engagement_pattern, churn_risk)
            triggers = self._identify_behavioral_triggers(lead_data, engagement_pattern)
            risk_factors = self._identify_risk_factors(lead_data, churn_risk)
            opportunities = self._identify_opportunities(lead_data, engagement_score)
            
            return BehavioralInsight(
                lead_id=lead_data.get('lead_id', 'unknown'),
                next_best_action=next_action,
                action_confidence=action_confidence,
                churn_risk=churn_risk,
                churn_probability=churn_probability,
                engagement_pattern=engagement_pattern,
                engagement_score=engagement_score,
                predicted_conversion_timeline=predicted_timeline,
                recommended_touchpoints=touchpoints,
                behavioral_triggers=triggers,
                risk_factors=risk_factors,
                opportunity_indicators=opportunities
            )
            
        except Exception as e:
            logger.error(f"Error predicting behavioral insights: {e}")
            # Return default insight
            return BehavioralInsight(
                lead_id=lead_data.get('lead_id', 'unknown'),
                next_best_action=NextBestAction.EMAIL_FOLLOW_UP,
                action_confidence=0.5,
                churn_risk=ChurnRiskLevel.MEDIUM,
                churn_probability=0.3,
                engagement_pattern=EngagementPattern.MODERATELY_ENGAGED,
                engagement_score=0.6,
                predicted_conversion_timeline=30,
                recommended_touchpoints=['email', 'phone'],
                behavioral_triggers=['website_visit'],
                risk_factors=['low_engagement'],
                opportunity_indicators=['recent_activity']
            )
    
    