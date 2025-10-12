from typing import Dict, List, Any
import json
import pandas as pd
from models.meta_lead_generation.model import MetaLeadGenerationModel, LeadGenerationResult
from .data_sources import RealTimeDataIntegrator

class MetaLeadGenerationInference:
    """
    Enhanced inference wrapper for the Meta Lead Generation Model
    """
    
    def __init__(self, model_config: Dict[str, str] = None):
        self.model = MetaLeadGenerationModel(model_config or {})
        self.data_integrator = None
        self._initialize_data_integration()
    
    def _initialize_data_integration(self):
        """Initialize real-time data integration"""
        try:
            self.data_integrator = RealTimeDataIntegrator()
            logger.info("Real-time data integration initialized")
        except Exception as e:
            logger.warning(f"Data integration initialization failed: {e}")
            self.data_integrator = None
    
    async def score_lead_with_enrichment(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score lead with real-time data enrichment"""
        try:
            # Enrich lead data if integrator is available
            if self.data_integrator:
                async with self.data_integrator as integrator:
                    enriched_data = await integrator.enrich_lead_data(lead_data)
            else:
                enriched_data = lead_data
            
            # Score the enriched lead
            scoring_result = self.score_lead(enriched_data)
            
            # Add enrichment insights
            if 'social_signals' in enriched_data:
                scoring_result['social_insights'] = self._analyze_social_signals(
                    enriched_data['social_signals'])
            
            if 'financial_indicators' in enriched_data:
                scoring_result['financial_insights'] = self._analyze_financial_indicators(
                    enriched_data['financial_indicators'])
            
            if 'market_context' in enriched_data:
                scoring_result['market_insights'] = self._analyze_market_context(
                    enriched_data['market_context'])
            
            return scoring_result
            
        except Exception as e:
            logger.error(f"Error in enriched lead scoring: {e}")
            # Fallback to regular scoring
            return self.score_lead(lead_data)
    
    def _analyze_social_signals(self, social_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social media signals for insights"""
        insights = {
            'social_engagement_level': 'low',
            'professional_presence': social_signals.get('professional_presence', False),
            'influence_tier': 'standard',
            'social_score_adjustment': 0.0
        }
        
        engagement_score = social_signals.get('engagement_score', 0)
        influence_score = social_signals.get('influence_score', 0)
        
        # Determine engagement level
        if engagement_score > 7:
            insights['social_engagement_level'] = 'high'
            insights['social_score_adjustment'] = 0.15
        elif engagement_score > 4:
            insights['social_engagement_level'] = 'medium'
            insights['social_score_adjustment'] = 0.05
        
        # Determine influence tier
        if influence_score > 8:
            insights['influence_tier'] = 'influencer'
            insights['social_score_adjustment'] += 0.10
        elif influence_score > 5:
            insights['influence_tier'] = 'engaged_user'
            insights['social_score_adjustment'] += 0.05
        
        return insights
    
    def _analyze_financial_indicators(self, financial_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial indicators for insights"""
        insights = {
            'financial_stability': 'medium',
            'affordability_tier': 'standard',
            'risk_level': 'medium',
            'financial_score_adjustment': 0.0
        }
        
        affordability_score = financial_indicators.get('insurance_affordability_score', 5)
        income_stability = financial_indicators.get('income_stability', 5)
        
        # Assess financial stability
        if income_stability > 7 and affordability_score > 7:
            insights['financial_stability'] = 'high'
            insights['financial_score_adjustment'] = 0.20
        elif income_stability > 5 and affordability_score > 5:
            insights['financial_stability'] = 'medium'
            insights['financial_score_adjustment'] = 0.10
        else:
            insights['financial_stability'] = 'low'
            insights['financial_score_adjustment'] = -0.10
        
        # Determine affordability tier
        if affordability_score > 8:
            insights['affordability_tier'] = 'premium'
        elif affordability_score > 6:
            insights['affordability_tier'] = 'standard'
        else:
            insights['affordability_tier'] = 'budget'
        
        return insights
    
    def _analyze_market_context(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market context for timing insights"""
        insights = {
            'market_timing': 'neutral',
            'competitive_advantage': 'standard',
            'urgency_multiplier': 1.0,
            'market_score_adjustment': 0.0
        }
        
        seasonal_multiplier = market_context.get('seasonal_factors', {}).get('seasonal_multiplier', 1.0)
        competitive_landscape = market_context.get('competitive_landscape', {})
        
        # Assess market timing
        if seasonal_multiplier > 1.2:
            insights['market_timing'] = 'favorable'
            insights['urgency_multiplier'] = 1.3
            insights['market_score_adjustment'] = 0.10
        elif seasonal_multiplier < 0.9:
            insights['market_timing'] = 'challenging'
            insights['urgency_multiplier'] = 0.8
            insights['market_score_adjustment'] = -0.05
        
        # Assess competitive advantage
        avg_advantage = 0
        advantage_count = 0
        
        for product, data in competitive_landscape.items():
            if isinstance(data, dict) and 'rate_advantage' in data:
                avg_advantage += data['rate_advantage']
                advantage_count += 1
        
        if advantage_count > 0:
            avg_advantage /= advantage_count
            
            if avg_advantage > 0.10:
                insights['competitive_advantage'] = 'strong'
                insights['market_score_adjustment'] += 0.15
            elif avg_advantage > 0.05:
                insights['competitive_advantage'] = 'moderate'
                insights['market_score_adjustment'] += 0.05
        
        return insights
    
    def score_lead(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single lead across all insurance products with enhanced features"""
        result = self.model.generate_lead_score(lead_data)
        
        return {
            'lead_id': result.lead_id,
            'overall_score': result.overall_score,
            'product_scores': result.product_scores,
            'confidence_scores': result.confidence_scores,
            'market_adjusted_scores': result.market_adjusted_scores,
            'recommended_products': result.recommended_products,
            'priority_level': result.priority_level,
            'cross_sell_opportunities': result.cross_sell_opportunities,
            'lead_quality': result.lead_quality,
            'estimated_lifetime_value': result.estimated_lifetime_value,
            'next_best_action': result.next_best_action,
            'confidence_score': result.confidence_score,
            'conversion_velocity': result.conversion_velocity,
            'urgency_signals': result.urgency_signals,
            'optimal_contact_time': result.optimal_contact_time,
            'revenue_potential': result.revenue_potential,
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_version': 'meta_v2.0_enhanced'
        }
    
    def batch_score_leads(self, leads_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score multiple leads in batch"""
        results = self.model.batch_generate_leads(leads_data)
        
        return [
            {
                'lead_id': result.lead_id,
                'overall_score': result.overall_score,
                'product_scores': result.product_scores,
                'recommended_products': result.recommended_products,
                'priority_level': result.priority_level,
                'cross_sell_opportunities': result.cross_sell_opportunities,
                'lead_quality': result.lead_quality,
                'estimated_lifetime_value': result.estimated_lifetime_value,
                'next_best_action': result.next_best_action,
                'confidence_score': result.confidence_score
            }
            for result in results
        ]
    
    def get_lead_recommendations(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced recommendations with velocity and urgency insights"""
        result = self.model.generate_lead_score(lead_data)
        
        return {
            'lead_id': lead_data.get('lead_id'),
            'executive_summary': {
                'overall_score': result.overall_score,
                'lead_quality': result.lead_quality,
                'priority_level': result.priority_level,
                'revenue_potential': result.revenue_potential,
                'confidence_score': result.confidence_score
            },
            'product_analysis': {
                'raw_scores': result.product_scores,
                'market_adjusted_scores': result.market_adjusted_scores,
                'confidence_by_product': result.confidence_scores,
                'conversion_velocity': result.conversion_velocity
            },
            'urgency_analysis': {
                'urgency_signals': result.urgency_signals,
                'optimal_contact_time': result.optimal_contact_time,
                'priority_justification': self._explain_priority(result)
            },
            'sales_strategy': {
                'primary_products': result.recommended_products,
                'cross_sell_opportunities': result.cross_sell_opportunities,
                'next_best_action': result.next_best_action,
                'talking_points': self._generate_talking_points(result),
                'objection_handling': self._generate_objection_handling(result)
            },
            'follow_up_plan': self._create_enhanced_follow_up_plan(result),
            'competitive_analysis': self._analyze_competitive_position(result)
        }
    
    def _explain_priority(self, result: LeadGenerationResult) -> str:
        """Explain why lead received this priority level"""
        explanations = []
        
        if result.overall_score >= 85:
            explanations.append(f"High overall score ({result.overall_score:.1f})")
        
        if result.urgency_signals:
            explanations.append(f"Urgency signals detected: {', '.join(result.urgency_signals[:3])}")
        
        if result.revenue_potential > 20000:
            explanations.append(f"High revenue potential (${result.revenue_potential:,.0f})")
        
        if len(result.recommended_products) >= 2:
            explanations.append("Multiple product opportunities")
        
        return "; ".join(explanations) if explanations else "Standard scoring criteria"
    
    def _generate_talking_points(self, result: LeadGenerationResult) -> List[str]:
        """Generate sales talking points based on lead analysis"""
        talking_points = []
        
        # Primary product focus
        if result.recommended_products:
            primary = result.recommended_products[0]
            talking_points.append(f"Lead shows strong fit for {primary} insurance")
        
        # Urgency-based points
        if "OPEN_ENROLLMENT_PERIOD" in result.urgency_signals:
            talking_points.append("Time-sensitive: Open enrollment deadline approaching")
        
        if "NEW_BABY" in result.urgency_signals:
            talking_points.append("Congratulations on your new addition - let's protect your growing family")
        
        if "NO_CURRENT_COVERAGE" in result.urgency_signals:
            talking_points.append("Currently uninsured - immediate protection needed")
        
        # Cross-sell opportunities
        if result.cross_sell_opportunities:
            talking_points.append(f"Bundle opportunity with {result.cross_sell_opportunities[0]}")
        
        # Value proposition
        if result.revenue_potential > 15000:
            talking_points.append("High-value client - premium service approach")
        
        return talking_points
    
    def _generate_objection_handling(self, result: LeadGenerationResult) -> Dict[str, str]:
        """Generate objection handling strategies"""
        objections = {}
        
        # Price objections
        if result.revenue_potential > 10000:
            objections["price"] = "Focus on comprehensive coverage value and long-term savings"
        else:
            objections["price"] = "Emphasize affordable options and payment plans"
        
        # Timing objections
        if result.urgency_signals:
            objections["timing"] = f"Highlight urgency: {result.urgency_signals[0].replace('_', ' ').lower()}"
        else:
            objections["timing"] = "Emphasize peace of mind and protection benefits"
        
        # Need objections
        if "HIGH_HEALTH_COMPLEXITY" in result.urgency_signals:
            objections["need"] = "Address specific health concerns and coverage gaps"
        else:
            objections["need"] = "Focus on life changes and future planning"
        
        return objections
    
    def _create_enhanced_follow_up_plan(self, result: LeadGenerationResult) -> Dict[str, Any]:
        """Create detailed follow-up plan with timing and content"""
        plan = {
            'immediate_action': result.next_best_action,
            'optimal_contact_time': result.optimal_contact_time,
            'follow_up_sequence': []
        }
        
        # Determine follow-up sequence based on velocity
        primary_product = result.recommended_products[0] if result.recommended_products else 'general'
        velocity = result.conversion_velocity.get(primary_product, 'MEDIUM')
        
        if velocity == 'IMMEDIATE':
            plan['follow_up_sequence'] = [
                {'timing': '2_hours', 'action': 'Phone call', 'content': 'Urgent quote discussion'},
                {'timing': '1_day', 'action': 'Email quote', 'content': 'Detailed proposal'},
                {'timing': '3_days', 'action': 'Follow-up call', 'content': 'Address questions'}
            ]
        elif velocity == 'FAST':
            plan['follow_up_sequence'] = [
                {'timing': '4_hours', 'action': 'Email introduction', 'content': 'Personalized quote'},
                {'timing': '2_days', 'action': 'Phone call', 'content': 'Discuss options'},
                {'timing': '1_week', 'action': 'Follow-up email', 'content': 'Additional options'}
            ]
        elif velocity == 'MEDIUM':
            plan['follow_up_sequence'] = [
                {'timing': '1_day', 'action': 'Email nurture', 'content': 'Educational content'},
                {'timing': '1_week', 'action': 'Phone call', 'content': 'Needs assessment'},
                {'timing': '2_weeks', 'action': 'Email quote', 'content': 'Customized proposal'}
            ]
        else:  # SLOW
            plan['follow_up_sequence'] = [
                {'timing': '3_days', 'action': 'Email series', 'content': 'Educational campaign'},
                {'timing': '2_weeks', 'action': 'Phone call', 'content': 'Relationship building'},
                {'timing': '1_month', 'action': 'Email check-in', 'content': 'Needs update'}
            ]
        
        return plan
    
    def _analyze_competitive_position(self, result: LeadGenerationResult) -> Dict[str, Any]:
        """Analyze competitive positioning for this lead"""
        analysis = {
            'competitive_risk': 'LOW',
            'differentiation_strategy': [],
            'win_probability': result.confidence_score
        }
        
        # Assess competitive risk
        if result.overall_score > 80 and result.lead_quality in ['HIGH', 'PREMIUM']:
            analysis['competitive_risk'] = 'HIGH'
            analysis['differentiation_strategy'].append('Emphasize superior service and expertise')
        
        if len(result.recommended_products) > 1:
            analysis['differentiation_strategy'].append('Highlight comprehensive multi-product solution')
        
        if result.urgency_signals:
            analysis['competitive_risk'] = 'MEDIUM'
            analysis['differentiation_strategy'].append('Leverage timing advantage and urgency')
        
        # Adjust win probability based on competitive factors
        if analysis['competitive_risk'] == 'HIGH':
            analysis['win_probability'] *= 0.8
        elif analysis['competitive_risk'] == 'LOW':
            analysis['win_probability'] *= 1.1
        
        analysis['win_probability'] = min(1.0, analysis['win_probability'])
        
        return analysis
    
    def update_market_conditions(self, market_updates: Dict[str, float]):
        """Update market demand multipliers"""
        self.model.update_market_multipliers(market_updates)
        return {"status": "success", "updated_multipliers": self.model.market_multipliers}
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get current model performance and configuration"""
        return {
            'model_version': 'meta_v2.0_enhanced',
            'market_multipliers': self.model.market_multipliers,
            'quality_thresholds': self.model.quality_thresholds,
            'priority_thresholds': self.model.priority_thresholds,
            'velocity_thresholds': self.model.velocity_thresholds,
            'revenue_multipliers': self.model.revenue_multipliers,
            'features': [
                'confidence_weighted_scoring',
                'market_demand_adjustment',
                'urgency_signal_detection',
                'conversion_velocity_prediction',
                'optimal_contact_timing',
                'revenue_potential_calculation',
                'enhanced_cross_sell_identification'
            ]
        }

