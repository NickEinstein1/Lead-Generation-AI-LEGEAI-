"""
Ensemble Scorer for Health Insurance Leads
Combines XGBoost and Deep Learning models with adaptive weighting
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from backend.models.insurance_lead_scoring.inference import InsuranceLeadScorer
from backend.models.healthcare_insurance_scoring.inference_deep_learning import HealthInsuranceDeepLearningScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleHealthInsuranceScorer:
    """
    Ensemble scorer combining XGBoost and Deep Learning for health insurance
    Uses adaptive weighting based on model confidence
    """
    
    def __init__(self):
        self.xgboost_scorer = None
        self.deep_learning_scorer = None
        
        # Default weights (can be adjusted based on confidence)
        self.default_xgboost_weight = 0.4
        self.default_dl_weight = 0.6
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize both XGBoost and Deep Learning models"""
        try:
            # Initialize XGBoost scorer (general insurance model)
            self.xgboost_scorer = InsuranceLeadScorer()
            logger.info("XGBoost model loaded for health insurance ensemble")
        except Exception as e:
            logger.warning(f"Could not load XGBoost model: {e}")

        try:
            # Initialize Deep Learning scorer (health-specific)
            self.deep_learning_scorer = HealthInsuranceDeepLearningScorer()
            logger.info("Deep Learning model loaded for health insurance ensemble")
        except Exception as e:
            logger.warning(f"Could not load Deep Learning model: {e}")
    
    def _calculate_adaptive_weights(self, xgboost_confidence: float, dl_confidence: float) -> Dict[str, float]:
        """
        Calculate adaptive weights based on model confidence
        Higher confidence models get more weight
        """
        total_confidence = xgboost_confidence + dl_confidence
        
        if total_confidence == 0:
            return {
                'xgboost': self.default_xgboost_weight,
                'deep_learning': self.default_dl_weight
            }
        
        # Normalize confidences to get weights
        xgboost_weight = xgboost_confidence / total_confidence
        dl_weight = dl_confidence / total_confidence
        
        return {
            'xgboost': round(xgboost_weight, 3),
            'deep_learning': round(dl_weight, 3)
        }
    
    def score_lead(self, lead_data: Dict) -> Dict[str, Any]:
        """
        Score a health insurance lead using ensemble of models

        Args:
            lead_data: Dictionary containing lead information

        Returns:
            Dictionary with ensemble score and individual model scores
        """
        results = {
            'lead_id': lead_data.get('lead_id', 'unknown'),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_type': 'ensemble',
            'models_used': []
        }

        # Clean lead data - remove lead_id but keep consent fields for validation
        # The models' preprocessing will handle selecting only the actual features
        clean_data = {k: v for k, v in lead_data.items()
                     if k not in ['lead_id']}

        # Get predictions from both models
        xgboost_result = None
        dl_result = None

        if self.xgboost_scorer:
            try:
                xgboost_result = self.xgboost_scorer.score_lead(clean_data)
                results['models_used'].append('xgboost')
                results['xgboost_score'] = xgboost_result.get('score', 50.0)
                results['xgboost_confidence'] = xgboost_result.get('confidence', 0.5)
            except Exception as e:
                logger.error(f"XGBoost scoring failed: {e}")

        if self.deep_learning_scorer:
            try:
                dl_result = self.deep_learning_scorer.score_lead(clean_data)
                results['models_used'].append('deep_learning')
                results['deep_learning_score'] = dl_result.get('score', 50.0)
                results['deep_learning_confidence'] = dl_result.get('confidence', 0.5)
            except Exception as e:
                logger.error(f"Deep Learning scoring failed: {e}")
        
        # Calculate ensemble score
        if xgboost_result and dl_result:
            # Both models available - use ensemble
            xgb_score = results['xgboost_score']
            xgb_conf = results['xgboost_confidence']
            dl_score = results['deep_learning_score']
            dl_conf = results['deep_learning_confidence']
            
            # Calculate adaptive weights
            weights = self._calculate_adaptive_weights(xgb_conf, dl_conf)
            results['ensemble_weights'] = weights
            
            # Calculate weighted ensemble score
            ensemble_score = (xgb_score * weights['xgboost']) + (dl_score * weights['deep_learning'])
            
            # Calculate ensemble confidence (weighted average)
            ensemble_confidence = (xgb_conf * weights['xgboost']) + (dl_conf * weights['deep_learning'])
            
            results['score'] = round(ensemble_score, 2)
            results['confidence'] = round(ensemble_confidence, 3)
            results['model_version'] = None
            results['error'] = None
            
        elif xgboost_result:
            # Only XGBoost available
            results['score'] = results['xgboost_score']
            results['confidence'] = results['xgboost_confidence']
            results['model_version'] = '1.0_xgboost_only'
            results['error'] = 'Deep Learning model not available'
            
        elif dl_result:
            # Only Deep Learning available
            results['score'] = results['deep_learning_score']
            results['confidence'] = results['deep_learning_confidence']
            results['model_version'] = '1.0_dl_only'
            results['error'] = 'XGBoost model not available'
            
        else:
            # No models available
            results['score'] = 0
            results['confidence'] = 0.0
            results['error'] = 'No models available'
            results['model_version'] = None
        
        return results

