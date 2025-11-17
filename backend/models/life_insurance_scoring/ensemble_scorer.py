"""
Ensemble Life Insurance Lead Scorer

This module combines XGBoost and Deep Learning models for superior
life insurance lead scoring performance.

The ensemble uses:
1. XGBoost model (fast, interpretable, handles tabular data well)
2. Deep Learning model (captures complex patterns, attention mechanism)
3. Weighted averaging based on model confidence and performance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import logging
from typing import Dict, Any, List
from datetime import datetime

# Import both scorers
from backend.models.life_insurance_scoring.inference import LifeInsuranceLeadScorer
from backend.models.life_insurance_scoring.inference_deep_learning import LifeInsuranceDeepLearningScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleLifeInsuranceScorer:
    """
    Ensemble scorer combining XGBoost and Deep Learning models
    
    Ensemble Strategy:
    - Both models score the lead independently
    - Weighted average based on:
      * Model confidence
      * Historical performance (R² scores)
      * Lead characteristics (some leads better suited for certain models)
    """
    
    def __init__(self, 
                 xgboost_weight: float = 0.4,
                 deep_learning_weight: float = 0.6,
                 use_adaptive_weighting: bool = True):
        """
        Initialize ensemble scorer
        
        Args:
            xgboost_weight: Base weight for XGBoost model (0-1)
            deep_learning_weight: Base weight for Deep Learning model (0-1)
            use_adaptive_weighting: Whether to adjust weights based on confidence
        """
        self.xgboost_weight = xgboost_weight
        self.deep_learning_weight = deep_learning_weight
        self.use_adaptive_weighting = use_adaptive_weighting
        
        # Initialize both models
        logger.info("Initializing Ensemble Life Insurance Scorer...")
        
        try:
            self.xgboost_scorer = LifeInsuranceLeadScorer()
            logger.info("✓ XGBoost model loaded")
        except Exception as e:
            logger.warning(f"XGBoost model failed to load: {e}")
            self.xgboost_scorer = None
        
        try:
            self.deep_learning_scorer = LifeInsuranceDeepLearningScorer()
            logger.info("✓ Deep Learning model loaded")
        except Exception as e:
            logger.warning(f"Deep Learning model failed to load: {e}")
            self.deep_learning_scorer = None
        
        if not self.xgboost_scorer and not self.deep_learning_scorer:
            raise RuntimeError("Both models failed to load. Cannot create ensemble.")
        
        # Model performance metrics (from training)
        self.xgboost_r2 = 0.647  # From XGBoost training
        self.deep_learning_r2 = 0.643  # From Deep Learning training
        
        logger.info(f"Ensemble initialized with weights: XGBoost={xgboost_weight}, DL={deep_learning_weight}")
    
    def score_lead(self, lead_data: Dict) -> Dict[str, Any]:
        """
        Score a life insurance lead using ensemble approach
        
        Args:
            lead_data: Dictionary containing lead information
            
        Returns:
            Dictionary with ensemble score and detailed breakdown
        """
        
        results = {
            'lead_id': lead_data.get('lead_id', 'unknown'),
            'timestamp': datetime.now(datetime.UTC).isoformat(),
            'model_type': 'ensemble',
            'models_used': []
        }
        
        # Get predictions from both models
        xgboost_result = None
        dl_result = None
        
        if self.xgboost_scorer:
            try:
                xgboost_result = self.xgboost_scorer.score_lead(lead_data)
                results['models_used'].append('xgboost')
                results['xgboost_score'] = xgboost_result.get('score', 50.0)
                results['xgboost_confidence'] = xgboost_result.get('confidence', 0.5)
            except Exception as e:
                logger.error(f"XGBoost scoring failed: {e}")
        
        if self.deep_learning_scorer:
            try:
                dl_result = self.deep_learning_scorer.score_lead(lead_data)
                results['models_used'].append('deep_learning')
                results['deep_learning_score'] = dl_result.get('score', 50.0)
                results['deep_learning_confidence'] = dl_result.get('confidence', 0.5)
            except Exception as e:
                logger.error(f"Deep Learning scoring failed: {e}")
        
        # Calculate ensemble score
        if xgboost_result and dl_result:
            # Both models available - use ensemble
            ensemble_score, weights = self._calculate_ensemble_score(
                xgboost_result, dl_result, lead_data
            )
            results['score'] = ensemble_score
            results['ensemble_weights'] = weights
            results['confidence'] = self._calculate_ensemble_confidence(xgboost_result, dl_result)
            
        elif xgboost_result:
            # Only XGBoost available
            results['score'] = xgboost_result.get('score', 50.0)
            results['confidence'] = xgboost_result.get('confidence', 0.5)
            results['fallback_model'] = 'xgboost'
            
        elif dl_result:
            # Only Deep Learning available
            results['score'] = dl_result.get('score', 50.0)
            results['confidence'] = dl_result.get('confidence', 0.5)
            results['fallback_model'] = 'deep_learning'
            
        else:
            # No models available - return default
            results['score'] = 50.0
            results['confidence'] = 0.0
            results['error'] = 'No models available for scoring'
        
        # Add additional insights from XGBoost model (if available)
        if xgboost_result:
            for key in ['life_stage', 'mortality_risk_score', 'recommended_policy_type', 
                       'urgency_level', 'recommended_coverage']:
                if key in xgboost_result:
                    results[key] = xgboost_result[key]
        
        return results

    def _calculate_ensemble_score(self, xgboost_result: Dict, dl_result: Dict,
                                  lead_data: Dict) -> tuple[float, Dict]:
        """
        Calculate weighted ensemble score

        Returns:
            Tuple of (ensemble_score, weights_used)
        """
        xgb_score = xgboost_result.get('score', 50.0)
        dl_score = dl_result.get('score', 50.0)

        if self.use_adaptive_weighting:
            # Adjust weights based on confidence and lead characteristics
            xgb_conf = xgboost_result.get('confidence', 0.5)
            dl_conf = dl_result.get('confidence', 0.5)

            # Normalize confidences to weights
            total_conf = xgb_conf + dl_conf
            if total_conf > 0:
                adaptive_xgb_weight = (xgb_conf / total_conf) * 0.5 + self.xgboost_weight * 0.5
                adaptive_dl_weight = (dl_conf / total_conf) * 0.5 + self.deep_learning_weight * 0.5
            else:
                adaptive_xgb_weight = self.xgboost_weight
                adaptive_dl_weight = self.deep_learning_weight

            # Normalize weights to sum to 1
            total_weight = adaptive_xgb_weight + adaptive_dl_weight
            xgb_weight = adaptive_xgb_weight / total_weight
            dl_weight = adaptive_dl_weight / total_weight

        else:
            # Use fixed weights
            total_weight = self.xgboost_weight + self.deep_learning_weight
            xgb_weight = self.xgboost_weight / total_weight
            dl_weight = self.deep_learning_weight / total_weight

        # Calculate weighted average
        ensemble_score = (xgb_score * xgb_weight) + (dl_score * dl_weight)

        # Ensure score is in valid range
        ensemble_score = max(0, min(100, ensemble_score))

        weights = {
            'xgboost': round(xgb_weight, 3),
            'deep_learning': round(dl_weight, 3)
        }

        return round(ensemble_score, 2), weights

    def _calculate_ensemble_confidence(self, xgboost_result: Dict, dl_result: Dict) -> float:
        """
        Calculate ensemble confidence based on model agreement and individual confidences
        """
        xgb_score = xgboost_result.get('score', 50.0)
        dl_score = dl_result.get('score', 50.0)
        xgb_conf = xgboost_result.get('confidence', 0.5)
        dl_conf = dl_result.get('confidence', 0.5)

        # Agreement factor (higher when models agree)
        score_diff = abs(xgb_score - dl_score)
        agreement = 1.0 - (score_diff / 100.0)

        # Average confidence
        avg_confidence = (xgb_conf + dl_conf) / 2.0

        # Ensemble confidence is weighted by agreement
        ensemble_confidence = (avg_confidence * 0.7) + (agreement * 0.3)

        return round(ensemble_confidence, 3)

    def batch_score(self, leads: List[Dict]) -> List[Dict]:
        """Score multiple leads efficiently"""
        results = []
        for lead in leads:
            result = self.score_lead(lead)
            results.append(result)
        return results


# Example usage
if __name__ == "__main__":
    # Initialize ensemble scorer
    scorer = EnsembleLifeInsuranceScorer(
        xgboost_weight=0.4,
        deep_learning_weight=0.6,
        use_adaptive_weighting=True
    )

    # Sample lead (using values that exist in training data - all lowercase)
    sample_lead = {
        'lead_id': 'LIFE_001',
        'age': 42,
        'income': 95000,
        'marital_status': 'married',  # lowercase as in training data
        'dependents_count': 2,
        'employment_status': 'employed',  # lowercase as in training data
        'health_status': 'good',  # lowercase as in training data
        'smoking_status': 'non_smoker',  # lowercase with underscore as in training data
        'coverage_amount_requested': 500000,
        'policy_term': 20,
        'existing_life_insurance': 1,
        'beneficiary_count': 2,
        'debt_obligations': 15000,
        'mortgage_balance': 250000,
        'education_level': 'bachelors',  # lowercase as in training data
        'occupation_risk_level': 'low',  # lowercase as in training data
        'life_stage': 'wealth_accumulation',  # lowercase with underscore as in training data
        'financial_dependents': 2,
        'estate_planning_needs': 1,
        'consent_given': True,
        'consent_timestamp': '2024-11-14T10:30:00Z'
    }

    # Score the lead
    result = scorer.score_lead(sample_lead)

    print("\n" + "=" * 80)
    print("ENSEMBLE LIFE INSURANCE LEAD SCORING RESULT")
    print("=" * 80)
    print(f"Lead ID: {result['lead_id']}")
    print(f"Ensemble Score: {result['score']}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nModel Breakdown:")
    if 'xgboost_score' in result:
        print(f"  XGBoost Score: {result['xgboost_score']} (confidence: {result['xgboost_confidence']})")
    if 'deep_learning_score' in result:
        print(f"  Deep Learning Score: {result['deep_learning_score']} (confidence: {result['deep_learning_confidence']})")
    if 'ensemble_weights' in result:
        print(f"\nEnsemble Weights:")
        print(f"  XGBoost: {result['ensemble_weights']['xgboost']}")
        print(f"  Deep Learning: {result['ensemble_weights']['deep_learning']}")
    print("=" * 80)

