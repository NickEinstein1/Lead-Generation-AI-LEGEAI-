from typing import Dict, List, Any
import json
from models.meta_lead_generation.model import MetaLeadGenerationModel, LeadGenerationResult

class MetaLeadGenerationInference:
    """
    Inference wrapper for the Meta Lead Generation Model
    """
    
    def __init__(self, model_config: Dict[str, str] = None):
        self.model = MetaLeadGenerationModel(model_config or {})
        
    def score_lead(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single lead across all insurance products"""
        result = self.model.generate_lead_score(lead_data)
        
        return {
            'lead_id': result.lead_id,
            'overall_score': result.overall_score,
            'product_scores': result.product_scores,
            'recommended_products': result.recommended_products,
            'priority_level': result.priority_level,
            'cross_sell_opportunities': result.cross_sell_opportunities,
            'lead_quality': result.lead_quality,
            'estimated_lifetime_value': result.estimated_lifetime_value,
            'next_best_action': result.next_best_action,
            'confidence_score': result.confidence_score,
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_version': 'meta_v1.0'
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
        """Get detailed recommendations for a lead"""
        insights = self.model.get_lead_insights(lead_data)
        
        return {
            'lead_id': lead_data.get('lead_id'),
            'recommendations': insights,
            'action_plan': self._create_action_plan(insights),
            'follow_up_schedule': self._create_follow_up_schedule(insights)
        }
    
    def _create_action_plan(self, insights: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create actionable plan based on insights"""
        priority = insights['lead_summary']['priority']
        products = insights['recommendations']['primary_products']
        
        actions = []
        
        if priority == 'CRITICAL':
            actions.append({
                'action': 'immediate_call',
                'timeline': 'within_1_hour',
                'focus': f"Present {products[0] if products else 'insurance'} options"
            })
        elif priority == 'HIGH':
            actions.append({
                'action': 'priority_email',
                'timeline': 'within_4_hours',
                'focus': f"Send personalized {products[0] if products else 'insurance'} quote"
            })
        
        return actions
    
    def _create_follow_up_schedule(self, insights: Dict[str, Any]) -> Dict[str, str]:
        """Create follow-up schedule based on priority"""
        priority = insights['lead_summary']['priority']
        
        schedules = {
            'CRITICAL': {'first_follow_up': '1_hour', 'second_follow_up': '1_day'},
            'HIGH': {'first_follow_up': '4_hours', 'second_follow_up': '3_days'},
            'MEDIUM': {'first_follow_up': '1_day', 'second_follow_up': '1_week'},
            'LOW': {'first_follow_up': '1_week', 'second_follow_up': '1_month'}
        }
        
        return schedules.get(priority, schedules['LOW'])