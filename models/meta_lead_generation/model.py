import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import joblib
from dataclasses import dataclass
from enum import Enum

from models.insurance_lead_scoring.inference import InsuranceLeadScorer
from models.healthcare_insurance_scoring.inference import HealthcareInsuranceScorer
from models.life_insurance_scoring.inference import LifeInsuranceScorer

class InsuranceProduct(Enum):
    BASE = "base"
    HEALTHCARE = "healthcare"
    LIFE = "life"
    ALL = "all"

@dataclass
class LeadGenerationResult:
    lead_id: str
    overall_score: float
    product_scores: Dict[str, float]
    recommended_products: List[str]
    priority_level: str
    cross_sell_opportunities: List[str]
    lead_quality: str
    estimated_lifetime_value: float
    next_best_action: str
    confidence_score: float

class MetaLeadGenerationModel:
    """
    Meta model that orchestrates all insurance lead scoring models
    to provide comprehensive lead generation and prioritization
    """
    
    def __init__(self, model_paths: Dict[str, str]):
        self.base_scorer = InsuranceLeadScorer()
        self.healthcare_scorer = HealthcareInsuranceScorer()
        self.life_scorer = LifeInsuranceScorer()
        
        # Load meta-model weights and thresholds
        self.product_weights = {
            'base': 0.3,
            'healthcare': 0.35,
            'life': 0.35
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'HIGH': 80,
            'MEDIUM': 60,
            'LOW': 40
        }
        
        # Priority thresholds
        self.priority_thresholds = {
            'CRITICAL': 85,
            'HIGH': 70,
            'MEDIUM': 55,
            'LOW': 40
        }
    
    def generate_lead_score(self, lead_data: Dict[str, Any]) -> LeadGenerationResult:
        """
        Generate comprehensive lead score using all models
        """
        # Get scores from all models
        product_scores = self._get_all_product_scores(lead_data)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(product_scores)
        
        # Determine recommended products
        recommended_products = self._get_recommended_products(product_scores)
        
        # Assess priority level
        priority_level = self._assess_priority_level(overall_score, product_scores)
        
        # Identify cross-sell opportunities
        cross_sell_opportunities = self._identify_cross_sell_opportunities(product_scores)
        
        # Determine lead quality
        lead_quality = self._assess_lead_quality(overall_score)
        
        # Estimate lifetime value
        estimated_ltv = self._estimate_lifetime_value(product_scores, lead_data)
        
        # Determine next best action
        next_best_action = self._determine_next_action(
            overall_score, recommended_products, priority_level
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(product_scores)
        
        return LeadGenerationResult(
            lead_id=lead_data.get('lead_id', 'UNKNOWN'),
            overall_score=overall_score,
            product_scores=product_scores,
            recommended_products=recommended_products,
            priority_level=priority_level,
            cross_sell_opportunities=cross_sell_opportunities,
            lead_quality=lead_quality,
            estimated_lifetime_value=estimated_ltv,
            next_best_action=next_best_action,
            confidence_score=confidence_score
        )
    
    def _get_all_product_scores(self, lead_data: Dict[str, Any]) -> Dict[str, float]:
        """Get scores from all product models"""
        scores = {}
        
        try:
            # Base insurance score
            base_result = self.base_scorer.score_lead(lead_data)
            scores['base'] = base_result.get('score', 0)
        except Exception as e:
            scores['base'] = 0
            
        try:
            # Healthcare insurance score
            healthcare_result = self.healthcare_scorer.score_lead(lead_data)
            scores['healthcare'] = healthcare_result.get('score', 0)
        except Exception as e:
            scores['healthcare'] = 0
            
        try:
            # Life insurance score
            life_result = self.life_scorer.score_lead(lead_data)
            scores['life'] = life_result.get('score', 0)
        except Exception as e:
            scores['life'] = 0
            
        return scores
    
    def _calculate_overall_score(self, product_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        weighted_sum = sum(
            score * self.product_weights.get(product, 0)
            for product, score in product_scores.items()
        )
        
        # Apply ensemble boost for high-performing leads across multiple products
        high_scores = [score for score in product_scores.values() if score > 70]
        if len(high_scores) >= 2:
            ensemble_boost = min(5.0, len(high_scores) * 2.0)
            weighted_sum += ensemble_boost
            
        return min(100.0, weighted_sum)
    
    def _get_recommended_products(self, product_scores: Dict[str, float]) -> List[str]:
        """Determine which products to recommend based on scores"""
        recommendations = []
        
        # Sort products by score
        sorted_products = sorted(
            product_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for product, score in sorted_products:
            if score >= 60:  # Minimum threshold for recommendation
                recommendations.append(product)
                
        return recommendations
    
    def _assess_priority_level(self, overall_score: float, product_scores: Dict[str, float]) -> str:
        """Assess lead priority level"""
        max_product_score = max(product_scores.values()) if product_scores else 0
        
        # High priority if overall score is high OR any single product score is very high
        if overall_score >= self.priority_thresholds['CRITICAL'] or max_product_score >= 90:
            return 'CRITICAL'
        elif overall_score >= self.priority_thresholds['HIGH'] or max_product_score >= 80:
            return 'HIGH'
        elif overall_score >= self.priority_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _identify_cross_sell_opportunities(self, product_scores: Dict[str, float]) -> List[str]:
        """Identify cross-selling opportunities"""
        opportunities = []
        
        # If multiple products score well, identify cross-sell potential
        good_scores = {k: v for k, v in product_scores.items() if v >= 65}
        
        if len(good_scores) >= 2:
            # Sort by score and recommend top products for cross-selling
            sorted_products = sorted(good_scores.items(), key=lambda x: x[1], reverse=True)
            opportunities = [product for product, _ in sorted_products[1:]]  # Exclude top product
            
        return opportunities
    
    def _assess_lead_quality(self, overall_score: float) -> str:
        """Assess overall lead quality"""
        if overall_score >= self.quality_thresholds['HIGH']:
            return 'HIGH'
        elif overall_score >= self.quality_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _estimate_lifetime_value(self, product_scores: Dict[str, float], lead_data: Dict[str, Any]) -> float:
        """Estimate customer lifetime value based on scores and demographics"""
        base_ltv = {
            'base': 2500,
            'healthcare': 8000,
            'life': 15000
        }
        
        estimated_ltv = 0
        for product, score in product_scores.items():
            if score >= 60:  # Only count products likely to convert
                product_ltv = base_ltv.get(product, 0) * (score / 100)
                estimated_ltv += product_ltv
        
        # Apply demographic multipliers
        age = lead_data.get('age', 35)
        income = lead_data.get('income', 50000)
        
        # Age factor (peak earning years get higher LTV)
        age_factor = 1.0
        if 30 <= age <= 50:
            age_factor = 1.2
        elif age > 50:
            age_factor = 1.1
            
        # Income factor
        income_factor = min(2.0, income / 50000)
        
        return estimated_ltv * age_factor * income_factor
    
    def _determine_next_action(self, overall_score: float, recommended_products: List[str], 
                             priority_level: str) -> str:
        """Determine the next best action for this lead"""
        if priority_level == 'CRITICAL':
            return f"IMMEDIATE_CONTACT - Focus on {recommended_products[0] if recommended_products else 'base'} insurance"
        elif priority_level == 'HIGH':
            return f"PRIORITY_FOLLOW_UP - Present {', '.join(recommended_products[:2])}"
        elif priority_level == 'MEDIUM':
            return f"SCHEDULED_FOLLOW_UP - Nurture with {recommended_products[0] if recommended_products else 'general'} content"
        else:
            return "NURTURE_CAMPAIGN - Add to general marketing funnel"
    
    def _calculate_confidence(self, product_scores: Dict[str, float]) -> float:
        """Calculate confidence in the overall prediction"""
        if not product_scores:
            return 0.0
            
        # Higher confidence when scores are consistent across products
        scores = list(product_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Lower standard deviation = higher confidence
        consistency_factor = max(0.5, 1.0 - (std_score / 50.0))
        
        # Higher mean score = higher confidence
        score_factor = mean_score / 100.0
        
        return min(1.0, consistency_factor * score_factor)

    def batch_generate_leads(self, leads_data: List[Dict[str, Any]]) -> List[LeadGenerationResult]:
        """Process multiple leads in batch"""
        results = []
        for lead_data in leads_data:
            try:
                result = self.generate_lead_score(lead_data)
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = LeadGenerationResult(
                    lead_id=lead_data.get('lead_id', 'ERROR'),
                    overall_score=0.0,
                    product_scores={},
                    recommended_products=[],
                    priority_level='LOW',
                    cross_sell_opportunities=[],
                    lead_quality='LOW',
                    estimated_lifetime_value=0.0,
                    next_best_action='REVIEW_MANUALLY',
                    confidence_score=0.0
                )
                results.append(error_result)
                
        return results
    
    def get_lead_insights(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed insights about a lead"""
        result = self.generate_lead_score(lead_data)
        
        return {
            'lead_summary': {
                'overall_score': result.overall_score,
                'quality': result.lead_quality,
                'priority': result.priority_level,
                'estimated_ltv': result.estimated_lifetime_value
            },
            'product_breakdown': result.product_scores,
            'recommendations': {
                'primary_products': result.recommended_products,
                'cross_sell': result.cross_sell_opportunities,
                'next_action': result.next_best_action
            },
            'confidence_metrics': {
                'overall_confidence': result.confidence_score,
                'model_agreement': len([s for s in result.product_scores.values() if s > 60])
            }
        }