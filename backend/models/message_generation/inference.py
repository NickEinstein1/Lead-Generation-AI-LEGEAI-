from typing import Dict, List, Any, Optional
from backend.models.message_generation.model import MessageGenerationModel, MessageType, GeneratedMessage
import json

class MessageGenerationInference:
    """
    Inference wrapper for message generation based on lead scoring
    """
    
    def __init__(self):
        self.model = MessageGenerationModel()
    
    def generate_personalized_message(self, lead_data: Dict[str, Any],
                                    scoring_result: Dict[str, Any],
                                    message_type: str = "email") -> Dict[str, Any]:
        """Generate a single personalized message"""
        
        # Convert string to enum
        msg_type = MessageType(message_type.lower())
        
        # Generate message
        message = self.model.generate_message(lead_data, scoring_result, msg_type)
        
        return {
            "message_type": message.message_type,
            "subject_line": message.subject_line,
            "content": message.content,
            "call_to_action": message.call_to_action,
            "tone": message.tone,
            "urgency_level": message.urgency_level,
            "personalization_elements": message.personalization_elements,
            "follow_up_timing": message.follow_up_timing,
            "compliance_notes": message.compliance_notes,
            "generated_at": pd.Timestamp.now().isoformat()
        }
    
    def generate_message_sequence(self, lead_data: Dict[str, Any],
                                scoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete message sequence for a lead"""
        
        # Determine sequence based on conversion velocity
        velocity = scoring_result.get('conversion_velocity', {})
        primary_product = scoring_result.get('recommended_products', ['base'])[0]
        lead_velocity = velocity.get(primary_product, 'MEDIUM')
        
        sequence = []
        
        if lead_velocity == 'IMMEDIATE':
            # Immediate sequence: Email + SMS + Call
            sequence = [
                {"type": "email", "timing": "immediate", "purpose": "initial_contact"},
                {"type": "sms", "timing": "2_hours", "purpose": "urgency_reminder"},
                {"type": "call_script", "timing": "4_hours", "purpose": "direct_contact"}
            ]
        elif lead_velocity == 'FAST':
            # Fast sequence: Email + Follow-up + Call
            sequence = [
                {"type": "email", "timing": "immediate", "purpose": "initial_contact"},
                {"type": "email", "timing": "2_days", "purpose": "value_reinforcement"},
                {"type": "call_script", "timing": "5_days", "purpose": "consultation"}
            ]
        elif lead_velocity == 'MEDIUM':
            # Medium sequence: Educational approach
            sequence = [
                {"type": "email", "timing": "immediate", "purpose": "educational_content"},
                {"type": "email", "timing": "1_week", "purpose": "case_study"},
                {"type": "email", "timing": "2_weeks", "purpose": "consultation_offer"}
            ]
        else:  # SLOW
            # Slow sequence: Long-term nurture
            sequence = [
                {"type": "email", "timing": "immediate", "purpose": "welcome_education"},
                {"type": "email", "timing": "1_week", "purpose": "industry_insights"},
                {"type": "email", "timing": "3_weeks", "purpose": "soft_offer"},
                {"type": "email", "timing": "6_weeks", "purpose": "consultation_invite"}
            ]
        
        # Generate messages for each step
        generated_sequence = []
        for step in sequence:
            message = self.model.generate_message(
                lead_data, scoring_result, MessageType(step["type"])
            )
            
            generated_sequence.append({
                "step": len(generated_sequence) + 1,
                "timing": step["timing"],
                "purpose": step["purpose"],
                "message": {
                    "type": message.message_type,
                    "subject": message.subject_line,
                    "content": message.content,
                    "cta": message.call_to_action,
                    "tone": message.tone
                }
            })
        
        return {
            "lead_id": lead_data.get('lead_id'),
            "sequence_type": f"{lead_velocity.lower()}_velocity",
            "total_steps": len(generated_sequence),
            "sequence": generated_sequence,
            "estimated_conversion_time": self._estimate_conversion_time(lead_velocity),
            "success_probability": scoring_result.get('confidence_score', 0.5)
        }
    
    def generate_ab_test_variants(self, lead_data: Dict[str, Any],
                                scoring_result: Dict[str, Any],
                                message_type: str = "email",
                                num_variants: int = 3) -> Dict[str, Any]:
        """Generate A/B test variants for message optimization"""
        
        variants = []
        
        # Generate different variants by modifying strategy
        strategies = ["URGENT_ACTION_REQUIRED", "CONSULTATIVE_APPROACH", "EDUCATIONAL_NURTURE"]
        
        for i in range(min(num_variants, len(strategies))):
            # Temporarily modify scoring result for different strategy
            modified_scoring = scoring_result.copy()
            
            # Adjust priority for different strategies
            if strategies[i] == "URGENT_ACTION_REQUIRED":
                modified_scoring['priority_level'] = 'CRITICAL'
            elif strategies[i] == "EDUCATIONAL_NURTURE":
                modified_scoring['priority_level'] = 'MEDIUM'
            
            message = self.model.generate_message(
                lead_data, modified_scoring, MessageType(message_type.lower())
            )
            
            variants.append({
                "variant": f"Variant_{chr(65+i)}",  # A, B, C, etc.
                "strategy": strategies[i],
                "subject_line": message.subject_line,
                "content": message.content,
                "call_to_action": message.call_to_action,
                "tone": message.tone,
                "expected_performance": self._predict_variant_performance(strategies[i], scoring_result)
            })
        
        return {
            "lead_id": lead_data.get('lead_id'),
            "message_type": message_type,
            "variants": variants,
            "test_recommendation": self._recommend_test_split(variants),
            "success_metrics": ["open_rate", "click_rate", "response_rate", "conversion_rate"]
        }
    
    def _estimate_conversion_time(self, velocity: str) -> str:
        """Estimate time to conversion based on velocity"""
        time_estimates = {
            'IMMEDIATE': '24-48 hours',
            'FAST': '3-7 days',
            'MEDIUM': '2-4 weeks',
            'SLOW': '1-3 months'
        }
        return time_estimates.get(velocity, '2-4 weeks')
    
    def _predict_variant_performance(self, strategy: str, scoring_result: Dict[str, Any]) -> Dict[str, float]:
        """Predict performance metrics for message variants"""
        base_performance = {
            "open_rate": 0.25,
            "click_rate": 0.05,
            "response_rate": 0.02,
            "conversion_rate": 0.01
        }
        
        # Adjust based on strategy and lead quality
        lead_score = scoring_result.get('overall_score', 50)
        score_multiplier = lead_score / 50  # Normalize around 50
        
        strategy_multipliers = {
            "URGENT_ACTION_REQUIRED": {"open_rate": 1.3, "click_rate": 1.2, "response_rate": 1.4, "conversion_rate": 1.1},
            "CONSULTATIVE_APPROACH": {"open_rate": 1.1, "click_rate": 1.3, "response_rate": 1.2, "conversion_rate": 1.3},
            "EDUCATIONAL_NURTURE": {"open_rate": 1.2, "click_rate": 1.1, "response_rate": 0.9, "conversion_rate": 1.0}
        }
        
        multipliers = strategy_multipliers.get(strategy, {"open_rate": 1.0, "click_rate": 1.0, "response_rate": 1.0, "conversion_rate": 1.0})
        
        predicted_performance = {}
        for metric, base_rate in base_performance.items():
            predicted_performance[metric] = round(
                base_rate * multipliers[metric] * score_multiplier, 4
            )
        
        return predicted_performance
    
    def _recommend_test_split(self, variants: List[Dict]) -> Dict[str, Any]:
        """Recommend how to split A/B test traffic"""
        num_variants = len(variants)
        
        if num_variants == 2:
            return {"split": "50/50", "sample_size_per_variant": 100, "test_duration": "7 days"}
        elif num_variants == 3:
            return {"split": "40/30/30", "sample_size_per_variant": 75, "test_duration": "10 days"}
        else:
            equal_split = 100 // num_variants
            return {"split": f"{equal_split}% each", "sample_size_per_variant": 50, "test_duration": "14 days"}
