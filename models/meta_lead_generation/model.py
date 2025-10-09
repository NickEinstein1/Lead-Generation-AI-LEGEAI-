import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import joblib
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

from models.insurance_lead_scoring.inference import InsuranceLeadScorer
from models.healthcare_insurance_scoring.inference import HealthcareInsuranceLeadScorer
from models.life_insurance_scoring.inference import LifeInsuranceLeadScorer

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
    confidence_scores: Dict[str, float]
    market_adjusted_scores: Dict[str, float]
    recommended_products: List[str]
    priority_level: str
    cross_sell_opportunities: List[str]
    lead_quality: str
    estimated_lifetime_value: float
    next_best_action: str
    confidence_score: float
    conversion_velocity: Dict[str, str]
    urgency_signals: List[str]
    optimal_contact_time: str
    revenue_potential: float

class MetaLeadGenerationModel:
    """
    Enhanced Meta model that orchestrates all insurance lead scoring models
    with advanced lead generation capabilities
    """
    
    def __init__(self, model_paths: Dict[str, str] = None):
        self.base_scorer = InsuranceLeadScorer()
        self.healthcare_scorer = HealthcareInsuranceLeadScorer()
        self.life_scorer = LifeInsuranceLeadScorer()
        
        # Enhanced product weights with confidence factors
        self.product_weights = {
            'base': 0.3,
            'healthcare': 0.35,
            'life': 0.35
        }
        
        # Market demand multipliers (updated dynamically)
        self.market_multipliers = {
            'healthcare': 1.2,  # High demand season (open enrollment)
            'life': 1.1,        # Moderate demand
            'base': 0.95        # Lower priority this quarter
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'PREMIUM': 90,
            'HIGH': 75,
            'MEDIUM': 55,
            'LOW': 35
        }
        
        # Priority thresholds
        self.priority_thresholds = {
            'CRITICAL': 85,
            'HIGH': 70,
            'MEDIUM': 55,
            'LOW': 40
        }
        
        # Velocity thresholds
        self.velocity_thresholds = {
            'IMMEDIATE': 85,  # Convert within 24-48 hours
            'FAST': 70,       # Convert within 1-7 days
            'MEDIUM': 55,     # Convert within 1-4 weeks
            'SLOW': 40        # Convert within 1-3 months
        }
        
        # Revenue potential multipliers
        self.revenue_multipliers = {
            'base': 2500,
            'healthcare': 8000,
            'life': 15000
        }
    
    def generate_lead_score(self, lead_data: Dict[str, Any]) -> LeadGenerationResult:
        """
        Generate comprehensive lead score using enhanced ensemble approach
        """
        # Get scores and confidence from all models
        product_scores, confidence_scores = self._get_all_product_scores_with_confidence(lead_data)
        
        # Apply market demand adjustments
        market_adjusted_scores = self._apply_market_demand_boost(product_scores)
        
        # Calculate enhanced overall score with confidence weighting
        overall_score = self._calculate_weighted_ensemble_score(
            market_adjusted_scores, confidence_scores
        )
        
        # Detect urgency signals
        urgency_signals = self._detect_urgency_signals(lead_data)
        
        # Apply urgency boost
        if urgency_signals:
            urgency_boost = min(10.0, len(urgency_signals) * 2.5)
            overall_score = min(100.0, overall_score + urgency_boost)
        
        # Determine recommended products
        recommended_products = self._get_recommended_products(market_adjusted_scores)
        
        # Assess priority level
        priority_level = self._assess_priority_level(overall_score, market_adjusted_scores, urgency_signals)
        
        # Identify cross-sell opportunities
        cross_sell_opportunities = self._identify_cross_sell_opportunities(market_adjusted_scores)
        
        # Determine lead quality
        lead_quality = self._assess_lead_quality(overall_score)
        
        # Estimate lifetime value
        estimated_ltv = self._estimate_lifetime_value(market_adjusted_scores, lead_data)
        
        # Calculate conversion velocity
        conversion_velocity = self._calculate_conversion_velocity(lead_data, market_adjusted_scores)
        
        # Determine optimal contact time
        optimal_contact_time = self._determine_optimal_contact_time(lead_data, urgency_signals)
        
        # Calculate revenue potential
        revenue_potential = self._calculate_revenue_potential(market_adjusted_scores, lead_data)
        
        # Determine next best action
        next_best_action = self._determine_enhanced_next_action(
            overall_score, recommended_products, priority_level, conversion_velocity, urgency_signals
        )
        
        # Calculate overall confidence score
        confidence_score = self._calculate_overall_confidence(confidence_scores, product_scores)
        
        return LeadGenerationResult(
            lead_id=lead_data.get('lead_id', 'UNKNOWN'),
            overall_score=overall_score,
            product_scores=product_scores,
            confidence_scores=confidence_scores,
            market_adjusted_scores=market_adjusted_scores,
            recommended_products=recommended_products,
            priority_level=priority_level,
            cross_sell_opportunities=cross_sell_opportunities,
            lead_quality=lead_quality,
            estimated_lifetime_value=estimated_ltv,
            next_best_action=next_best_action,
            confidence_score=confidence_score,
            conversion_velocity=conversion_velocity,
            urgency_signals=urgency_signals,
            optimal_contact_time=optimal_contact_time,
            revenue_potential=revenue_potential
        )
    
    def _get_all_product_scores_with_confidence(self, lead_data: Dict[str, Any]) -> tuple:
        """Get scores and confidence from all product models"""
        scores = {}
        confidence_scores = {}
        
        try:
            # Base insurance score
            base_result = self.base_scorer.score_lead(lead_data)
            scores['base'] = base_result.get('score', 0)
            confidence_scores['base'] = self._calculate_model_confidence(base_result, 'base')
        except Exception as e:
            scores['base'] = 0
            confidence_scores['base'] = 0.1
            
        try:
            # Healthcare insurance score
            healthcare_result = self.healthcare_scorer.score_lead(lead_data)
            scores['healthcare'] = healthcare_result.get('score', 0)
            confidence_scores['healthcare'] = self._calculate_model_confidence(healthcare_result, 'healthcare')
        except Exception as e:
            scores['healthcare'] = 0
            confidence_scores['healthcare'] = 0.1
            
        try:
            # Life insurance score
            life_result = self.life_scorer.score_lead(lead_data)
            scores['life'] = life_result.get('score', 0)
            confidence_scores['life'] = self._calculate_model_confidence(life_result, 'life')
        except Exception as e:
            scores['life'] = 0
            confidence_scores['life'] = 0.1
            
        return scores, confidence_scores
    
    def _calculate_model_confidence(self, model_result: Dict, model_type: str) -> float:
        """Calculate confidence score for individual model predictions"""
        base_confidence = 0.7
        
        # Boost confidence based on model-specific factors
        if model_type == 'healthcare':
            # Higher confidence during open enrollment
            if model_result.get('open_enrollment_period', False):
                base_confidence += 0.2
            # Higher confidence for high health complexity
            if model_result.get('health_complexity_score', 0) > 5:
                base_confidence += 0.1
                
        elif model_type == 'life':
            # Higher confidence for family stage leads
            if model_result.get('life_stage') == 'family_building':
                base_confidence += 0.2
            # Higher confidence for high coverage requests
            if model_result.get('coverage_amount_requested', 0) > 500000:
                base_confidence += 0.1
                
        elif model_type == 'base':
            # Higher confidence for engaged leads
            if model_result.get('engagement_score', 0) > 7:
                base_confidence += 0.15
        
        return min(1.0, base_confidence)
    
    def _calculate_weighted_ensemble_score(self, product_scores: Dict[str, float], 
                                         confidence_scores: Dict[str, float]) -> float:
        """Calculate ensemble score weighted by model confidence"""
        total_weighted_score = 0
        total_weight = 0
        
        for product, score in product_scores.items():
            confidence = confidence_scores.get(product, 0.5)
            weight = self.product_weights[product] * confidence
            total_weighted_score += score * weight
            total_weight += weight
        
        base_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Apply ensemble boost for high-performing leads across multiple products
        high_scores = [score for score in product_scores.values() if score > 70]
        if len(high_scores) >= 2:
            ensemble_boost = min(8.0, len(high_scores) * 3.0)
            base_score += ensemble_boost
            
        return min(100.0, base_score)
    
    def _apply_market_demand_boost(self, product_scores: Dict[str, float]) -> Dict[str, float]:
        """Boost scores based on current market demand"""
        adjusted_scores = {}
        for product, score in product_scores.items():
            multiplier = self.market_multipliers.get(product, 1.0)
            adjusted_scores[product] = min(100, score * multiplier)
        
        return adjusted_scores
    
    def _detect_urgency_signals(self, lead_data: Dict[str, Any]) -> List[str]:
        """Detect urgency signals in lead data"""
        urgency_signals = []
        
        # Time-based urgency
        current_date = datetime.now()
        
        # Healthcare urgency signals
        if current_date.month == 11 or (current_date.month == 12 and current_date.day <= 15):
            urgency_signals.append("OPEN_ENROLLMENT_PERIOD")
        
        if lead_data.get('qualifying_life_event', False):
            urgency_signals.append("QUALIFYING_LIFE_EVENT")
        
        if lead_data.get('current_coverage') == 'none':
            urgency_signals.append("NO_CURRENT_COVERAGE")
        
        # Life insurance urgency signals
        if lead_data.get('recent_marriage', False):
            urgency_signals.append("RECENT_MARRIAGE")
        
        if lead_data.get('new_baby', False):
            urgency_signals.append("NEW_BABY")
        
        if lead_data.get('home_purchase', False):
            urgency_signals.append("HOME_PURCHASE")
        
        # Financial urgency signals
        if lead_data.get('income_increase', False):
            urgency_signals.append("INCOME_INCREASE")
        
        if lead_data.get('job_change', False):
            urgency_signals.append("JOB_CHANGE")
        
        # Health urgency signals
        if lead_data.get('health_conditions_count', 0) > 2:
            urgency_signals.append("HIGH_HEALTH_COMPLEXITY")
        
        # Age-based urgency
        age = lead_data.get('age', 35)
        if age >= 50 and not lead_data.get('has_life_insurance', False):
            urgency_signals.append("AGE_URGENCY")
        
        # Engagement urgency
        if lead_data.get('quote_requests_30d', 0) > 3:
            urgency_signals.append("HIGH_ENGAGEMENT")
        
        return urgency_signals
    
    def _calculate_conversion_velocity(self, lead_data: Dict[str, Any], 
                                     product_scores: Dict[str, float]) -> Dict[str, str]:
        """Predict how quickly lead might convert for each product"""
        velocity_predictions = {}
        
        # Base velocity factors
        age = lead_data.get('age', 35)
        income = lead_data.get('income', 50000)
        engagement = lead_data.get('engagement_score', 5)
        urgency_count = len(self._detect_urgency_signals(lead_data))
        
        for product, score in product_scores.items():
            # Calculate velocity score
            velocity_score = score
            
            # Age factor (younger people decide faster for life insurance)
            if product == 'life' and age < 40:
                velocity_score += 5
            elif product == 'healthcare' and age > 50:
                velocity_score += 5
            
            # Income factor (higher income = faster decisions)
            if income > 75000:
                velocity_score += 3
            elif income > 100000:
                velocity_score += 5
            
            # Engagement factor
            velocity_score += engagement
            
            # Urgency factor
            velocity_score += urgency_count * 3
            
            # Determine velocity category
            if velocity_score >= self.velocity_thresholds['IMMEDIATE']:
                velocity_predictions[product] = "IMMEDIATE"  # 24-48 hours
            elif velocity_score >= self.velocity_thresholds['FAST']:
                velocity_predictions[product] = "FAST"  # 1-7 days
            elif velocity_score >= self.velocity_thresholds['MEDIUM']:
                velocity_predictions[product] = "MEDIUM"  # 1-4 weeks
            else:
                velocity_predictions[product] = "SLOW"  # 1-3 months
        
        return velocity_predictions
    
    def _determine_optimal_contact_time(self, lead_data: Dict[str, Any], 
                                      urgency_signals: List[str]) -> str:
        """Determine the optimal time to contact the lead"""
        current_hour = datetime.now().hour
        
        # Immediate contact scenarios
        if any(signal in urgency_signals for signal in 
               ["OPEN_ENROLLMENT_PERIOD", "QUALIFYING_LIFE_EVENT", "NEW_BABY"]):
            return "IMMEDIATE"
        
        # High urgency - within business hours
        if any(signal in urgency_signals for signal in 
               ["NO_CURRENT_COVERAGE", "HIGH_ENGAGEMENT", "INCOME_INCREASE"]):
            if 9 <= current_hour <= 17:
                return "WITHIN_2_HOURS"
            else:
                return "NEXT_BUSINESS_MORNING"
        
        # Standard timing based on lead quality
        engagement = lead_data.get('engagement_score', 5)
        if engagement > 7:
            return "WITHIN_4_HOURS"
        elif engagement > 5:
            return "WITHIN_24_HOURS"
        else:
            return "WITHIN_3_DAYS"
    
    def _calculate_revenue_potential(self, product_scores: Dict[str, float], 
                                   lead_data: Dict[str, Any]) -> float:
        """Calculate total revenue potential across all products"""
        total_revenue = 0
        
        for product, score in product_scores.items():
            if score >= 50:  # Only count viable products
                base_revenue = self.revenue_multipliers.get(product, 0)
                
                # Apply score factor
                score_factor = score / 100
                
                # Apply demographic multipliers
                age = lead_data.get('age', 35)
                income = lead_data.get('income', 50000)
                
                # Age factor
                age_factor = 1.0
                if product == 'life' and 25 <= age <= 45:
                    age_factor = 1.3  # Prime life insurance age
                elif product == 'healthcare' and age > 50:
                    age_factor = 1.2  # Higher healthcare needs
                
                # Income factor
                income_factor = min(2.0, income / 50000)
                
                # Family factor
                family_size = lead_data.get('family_size', 1)
                family_factor = 1.0 + (family_size - 1) * 0.2
                
                product_revenue = base_revenue * score_factor * age_factor * income_factor * family_factor
                total_revenue += product_revenue
        
        return total_revenue
    
    def _determine_enhanced_next_action(self, overall_score: float, recommended_products: List[str], 
                                      priority_level: str, conversion_velocity: Dict[str, str],
                                      urgency_signals: List[str]) -> str:
        """Determine enhanced next best action with velocity and urgency considerations"""
        primary_product = recommended_products[0] if recommended_products else 'general'
        primary_velocity = conversion_velocity.get(primary_product, 'MEDIUM')
        
        # Critical priority with immediate velocity
        if priority_level == 'CRITICAL' and primary_velocity == 'IMMEDIATE':
            return f"URGENT_CALL_NOW - {primary_product.upper()} insurance, mention {', '.join(urgency_signals[:2])}"
        
        # Critical priority with fast velocity
        elif priority_level == 'CRITICAL':
            return f"PRIORITY_CALL_TODAY - {primary_product.upper()} focus, fast decision maker"
        
        # High priority with immediate velocity
        elif priority_level == 'HIGH' and primary_velocity == 'IMMEDIATE':
            return f"CALL_WITHIN_2_HOURS - {primary_product.upper()} + cross-sell opportunity"
        
        # High priority with fast velocity
        elif priority_level == 'HIGH':
            return f"CALL_TODAY - {primary_product.upper()} presentation with {recommended_products[1] if len(recommended_products) > 1 else 'add-ons'}"
        
        # Medium priority
        elif priority_level == 'MEDIUM':
            if primary_velocity in ['IMMEDIATE', 'FAST']:
                return f"EMAIL_QUOTE_TODAY - {primary_product.upper()} with follow-up call in 24h"
            else:
                return f"NURTURE_SEQUENCE - {primary_product.upper()} focused content series"
        
        # Low priority
        else:
            return f"MARKETING_AUTOMATION - General insurance education, monitor engagement"
    
    def _get_recommended_products(self, product_scores: Dict[str, float]) -> List[str]:
        """Determine which products to recommend based on market-adjusted scores"""
        recommendations = []
        
        # Sort products by market-adjusted score
        sorted_products = sorted(
            product_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for product, score in sorted_products:
            if score >= 55:  # Lowered threshold for market-adjusted scores
                recommendations.append(product)
                
        return recommendations
    
    def _assess_priority_level(self, overall_score: float, product_scores: Dict[str, float], 
                             urgency_signals: List[str]) -> str:
        """Enhanced priority assessment with urgency signals"""
        max_product_score = max(product_scores.values()) if product_scores else 0
        urgency_boost = len(urgency_signals) * 5
        
        adjusted_score = overall_score + urgency_boost
        
        if adjusted_score >= self.priority_thresholds['CRITICAL'] or max_product_score >= 90:
            return 'CRITICAL'
        elif adjusted_score >= self.priority_thresholds['HIGH'] or max_product_score >= 75:
            return 'HIGH'
        elif adjusted_score >= self.priority_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _identify_cross_sell_opportunities(self, product_scores: Dict[str, float]) -> List[str]:
        """Enhanced cross-sell identification with market adjustments"""
        opportunities = []
        
        # Lower threshold for market-adjusted scores
        good_scores = {k: v for k, v in product_scores.items() if v >= 60}
        
        if len(good_scores) >= 2:
            sorted_products = sorted(good_scores.items(), key=lambda x: x[1], reverse=True)
            opportunities = [product for product, _ in sorted_products[1:]]
            
        return opportunities
    
    def _assess_lead_quality(self, overall_score: float) -> str:
        """Enhanced lead quality assessment"""
        if overall_score >= self.quality_thresholds['PREMIUM']:
            return 'PREMIUM'
        elif overall_score >= self.quality_thresholds['HIGH']:
            return 'HIGH'
        elif overall_score >= self.quality_thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _estimate_lifetime_value(self, product_scores: Dict[str, float], lead_data: Dict[str, Any]) -> float:
        """Enhanced LTV calculation with market adjustments"""
        base_ltv = {
            'base': 2500,
            'healthcare': 8000,
            'life': 15000
        }
        
        estimated_ltv = 0
        for product, score in product_scores.items():
            if score >= 55:  # Adjusted threshold
                product_ltv = base_ltv.get(product, 0) * (score / 100)
                estimated_ltv += product_ltv
        
        # Enhanced demographic multipliers
        age = lead_data.get('age', 35)
        income = lead_data.get('income', 50000)
        family_size = lead_data.get('family_size', 1)
        
        # Age factor (more sophisticated)
        if 30 <= age <= 50:
            age_factor = 1.3
        elif 25 <= age <= 60:
            age_factor = 1.1
        else:
            age_factor = 0.9
            
        # Income factor (progressive)
        if income > 100000:
            income_factor = 2.0
        elif income > 75000:
            income_factor = 1.5
        else:
            income_factor = max(0.8, income / 50000)
        
        # Family factor
        family_factor = 1.0 + (family_size - 1) * 0.25
        
        return estimated_ltv * age_factor * income_factor * family_factor
    
    def _calculate_overall_confidence(self, confidence_scores: Dict[str, float], 
                                    product_scores: Dict[str, float]) -> float:
        """Calculate overall confidence in the meta prediction"""
        if not confidence_scores:
            return 0.0
        
        # Weight confidence by product scores
        weighted_confidence = 0
        total_weight = 0
        
        for product, confidence in confidence_scores.items():
            score = product_scores.get(product, 0)
            weight = score / 100  # Higher scores get more weight
            weighted_confidence += confidence * weight
            total_weight += weight
        
        base_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        # Boost confidence for consistent scores across products
        scores = list(product_scores.values())
        if len(scores) > 1:
            score_std = np.std(scores)
            consistency_boost = max(0, (50 - score_std) / 100)  # Lower std = higher boost
            base_confidence += consistency_boost * 0.2
        
        return min(1.0, base_confidence)

    def update_market_multipliers(self, new_multipliers: Dict[str, float]):
        """Update market demand multipliers dynamically"""
        self.market_multipliers.update(new_multipliers)
        logging.info(f"Updated market multipliers: {self.market_multipliers}")

    def batch_generate_leads(self, leads_data: List[Dict[str, Any]]) -> List[LeadGenerationResult]:
        """Process multiple leads in batch with enhanced features"""
        results = []
        for lead_data in leads_data:
            try:
                result = self.generate_lead_score(lead_data)
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing lead {lead_data.get('lead_id', 'UNKNOWN')}: {str(e)}")
                # Create error result
                error_result = LeadGenerationResult(
                    lead_id=lead_data.get('lead_id', 'ERROR'),
                    overall_score=0.0,
                    product_scores={},
                    confidence_scores={},
                    market_adjusted_scores={},
                    recommended_products=[],
                    priority_level='LOW',
                    cross_sell_opportunities=[],
                    lead_quality='LOW',
                    estimated_lifetime_value=0.0,
                    next_best_action='REVIEW_MANUALLY',
                    confidence_score=0.0,
                    conversion_velocity={},
                    urgency_signals=[],
                    optimal_contact_time='MANUAL_REVIEW',
                    revenue_potential=0.0
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
