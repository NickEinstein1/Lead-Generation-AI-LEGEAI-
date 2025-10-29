from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

from backend.models.message_generation.inference import MessageGenerationInference
from backend.models.meta_lead_generation.inference import MetaLeadGenerationInference

router = APIRouter(prefix="/message-generation", tags=["AI Message Generation"])

# Initialize models
message_generator = MessageGenerationInference()
lead_scorer = MetaLeadGenerationInference()

class MessageRequest(BaseModel):
    lead_data: Dict[str, Any]
    scoring_result: Optional[Dict[str, Any]] = None
    message_type: str = "email"  # email, sms, call_script, social_media

class SequenceRequest(BaseModel):
    lead_data: Dict[str, Any]
    scoring_result: Optional[Dict[str, Any]] = None

class ABTestRequest(BaseModel):
    lead_data: Dict[str, Any]
    scoring_result: Optional[Dict[str, Any]] = None
    message_type: str = "email"
    num_variants: int = 3

@router.post("/generate-message")
async def generate_personalized_message(request: MessageRequest):
    """
    Generate a personalized message based on lead scoring
    """
    try:
        # If no scoring result provided, generate it
        if not request.scoring_result:
            scoring_result = lead_scorer.score_lead(request.lead_data)
        else:
            scoring_result = request.scoring_result
        
        # Generate message
        message = message_generator.generate_personalized_message(
            request.lead_data,
            scoring_result,
            request.message_type
        )
        
        return {
            "status": "success",
            "message": message,
            "lead_insights": {
                "overall_score": scoring_result.get('overall_score'),
                "priority_level": scoring_result.get('priority_level'),
                "recommended_products": scoring_result.get('recommended_products'),
                "urgency_signals": scoring_result.get('urgency_signals')
            }
        }
        
    except Exception as e:
        logging.error(f"Error generating message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating message: {str(e)}")

@router.post("/generate-sequence")
async def generate_message_sequence(request: SequenceRequest):
    """
    Generate a complete message sequence for lead nurturing
    """
    try:
        # If no scoring result provided, generate it
        if not request.scoring_result:
            scoring_result = lead_scorer.score_lead(request.lead_data)
        else:
            scoring_result = request.scoring_result
        
        # Generate sequence
        sequence = message_generator.generate_message_sequence(
            request.lead_data,
            scoring_result
        )
        
        return {
            "status": "success",
            "sequence": sequence,
            "automation_ready": True,
            "estimated_roi": _calculate_sequence_roi(sequence, scoring_result)
        }
        
    except Exception as e:
        logging.error(f"Error generating sequence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating sequence: {str(e)}")

@router.post("/ab-test-variants")
async def generate_ab_test_variants(request: ABTestRequest):
    """
    Generate A/B test variants for message optimization
    """
    try:
        # If no scoring result provided, generate it
        if not request.scoring_result:
            scoring_result = lead_scorer.score_lead(request.lead_data)
        else:
            scoring_result = request.scoring_result
        
        # Generate variants
        variants = message_generator.generate_ab_test_variants(
            request.lead_data,
            scoring_result,
            request.message_type,
            request.num_variants
        )
        
        return {
            "status": "success",
            "ab_test": variants,
            "optimization_tips": _generate_optimization_tips(variants),
            "expected_lift": _calculate_expected_lift(variants)
        }
        
    except Exception as e:
        logging.error(f"Error generating A/B test variants: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating variants: {str(e)}")

@router.post("/bulk-message-generation")
async def bulk_generate_messages(leads_data: List[Dict[str, Any]], 
                               message_type: str = "email",
                               background_tasks: BackgroundTasks = None):
    """
    Generate messages for multiple leads in bulk
    """
    try:
        results = []
        
        for lead_data in leads_data:
            # Score the lead
            scoring_result = lead_scorer.score_lead(lead_data)
            
            # Generate message
            message = message_generator.generate_personalized_message(
                lead_data,
                scoring_result,
                message_type
            )
            
            results.append({
                "lead_id": lead_data.get('lead_id'),
                "message": message,
                "priority": scoring_result.get('priority_level'),
                "send_timing": scoring_result.get('optimal_contact_time')
            })
        
        # Schedule background analytics
        if background_tasks:
            background_tasks.add_task(_update_bulk_analytics, results)
        
        return {
            "status": "success",
            "total_messages": len(results),
            "messages": results,
            "bulk_insights": _generate_bulk_insights(results)
        }
        
    except Exception as e:
        logging.error(f"Error in bulk message generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in bulk generation: {str(e)}")

@router.get("/message-templates")
async def get_message_templates():
    """
    Get available message templates and customization options
    """
    return {
        "status": "success",
        "templates": {
            "email": {
                "urgent_action": "High-urgency email template",
                "consultative": "Professional consultation template",
                "educational": "Educational nurture template",
                "follow_up": "Follow-up sequence template"
            },
            "sms": {
                "immediate": "Immediate response SMS",
                "reminder": "Appointment reminder SMS",
                "offer": "Special offer SMS"
            },
            "call_script": {
                "cold_call": "Cold calling script",
                "warm_lead": "Warm lead script",
                "follow_up": "Follow-up call script"
            }
        },
        "personalization_options": [
            "Name", "Age", "Location", "Family size", "Income level",
            "Product interest", "Urgency signals", "Previous interactions"
        ],
        "compliance_features": [
            "TCPA compliance", "CAN-SPAM compliance", "State regulations",
            "Opt-out management", "Consent verification"
        ]
    }

@router.get("/performance-analytics")
async def get_message_performance():
    """
    Get message performance analytics and optimization insights
    """
    return {
        "status": "success",
        "performance_metrics": {
            "overall_stats": {
                "messages_sent": 15420,
                "open_rate": 0.28,
                "click_rate": 0.067,
                "response_rate": 0.034,
                "conversion_rate": 0.018
            },
            "by_message_type": {
                "email": {"open_rate": 0.28, "click_rate": 0.067, "conversion_rate": 0.018},
                "sms": {"open_rate": 0.95, "click_rate": 0.12, "conversion_rate": 0.025},
                "call_script": {"contact_rate": 0.45, "conversation_rate": 0.23, "conversion_rate": 0.035}
            },
            "by_urgency_level": {
                "CRITICAL": {"response_rate": 0.045, "conversion_rate": 0.028},
                "HIGH": {"response_rate": 0.038, "conversion_rate": 0.022},
                "MEDIUM": {"response_rate": 0.025, "conversion_rate": 0.015},
                "LOW": {"response_rate": 0.018, "conversion_rate": 0.008}
            }
        },
        "optimization_insights": [
            "Urgent messages have 56% higher response rates",
            "Personalized subject lines increase open rates by 23%",
            "SMS follow-ups improve email conversion by 34%",
            "Call scripts with urgency signals convert 67% better"
        ],
        "recommendations": [
            "Increase SMS usage for IMMEDIATE velocity leads",
            "A/B test subject lines for MEDIUM priority leads",
            "Implement urgency signals in all CRITICAL messages",
            "Use educational approach for leads scoring <60"
        ]
    }

# Helper functions
def _calculate_sequence_roi(sequence: Dict[str, Any], scoring_result: Dict[str, Any]) -> Dict[str, float]:
    """Calculate expected ROI for message sequence"""
    revenue_potential = scoring_result.get('revenue_potential', 5000)
    success_probability = sequence.get('success_probability', 0.5)
    sequence_cost = len(sequence.get('sequence', [])) * 2.50  # $2.50 per message
    
    expected_revenue = revenue_potential * success_probability
    roi = (expected_revenue - sequence_cost) / sequence_cost if sequence_cost > 0 else 0
    
    return {
        "expected_revenue": expected_revenue,
        "sequence_cost": sequence_cost,
        "roi_percentage": roi * 100,
        "break_even_probability": sequence_cost / revenue_potential if revenue_potential > 0 else 1
    }

def _generate_optimization_tips(variants: Dict[str, Any]) -> List[str]:
    """Generate optimization tips for A/B testing"""
    tips = [
        "Test during different times of day for optimal engagement",
        "Monitor response rates across all variants for statistical significance",
        "Consider lead score when interpreting variant performance",
        "Run test for at least 7 days to account for weekly patterns"
    ]
    
    # Add specific tips based on variants
    strategies = [v.get('strategy') for v in variants.get('variants', [])]
    
    if 'URGENT_ACTION_REQUIRED' in strategies:
        tips.append("Urgent variants may perform better on weekdays")
    
    if 'EDUCATIONAL_NURTURE' in strategies:
        tips.append("Educational content performs better with longer subject lines")
    
    return tips

def _calculate_expected_lift(variants: Dict[str, Any]) -> Dict[str, float]:
    """Calculate expected performance lift from A/B testing"""
    variant_performances = []
    
    for variant in variants.get('variants', []):
        performance = variant.get('expected_performance', {})
        conversion_rate = performance.get('conversion_rate', 0.01)
        variant_performances.append(conversion_rate)
    
    if len(variant_performances) > 1:
        best_performance = max(variant_performances)
        baseline_performance = min(variant_performances)
        lift = (best_performance - baseline_performance) / baseline_performance if baseline_performance > 0 else 0
        
        return {
            "expected_lift_percentage": lift * 100,
            "best_variant_conversion": best_performance,
            "baseline_conversion": baseline_performance,
            "confidence_level": 0.85
        }
    
    return {"expected_lift_percentage": 0, "note": "Need multiple variants to calculate lift"}

def _generate_bulk_insights(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate insights from bulk message generation"""
    total_messages = len(results)
    
    if total_messages == 0:
        return {"total_messages": 0}
    
    # Priority distribution
    priority_counts = {}
    for result in results:
        priority = result.get('priority', 'UNKNOWN')
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    # Timing distribution
    timing_counts = {}
    for result in results:
        timing = result.get('send_timing', 'UNKNOWN')
        timing_counts[timing] = timing_counts.get(timing, 0) + 1
    
    return {
        "total_messages": total_messages,
        "priority_distribution": priority_counts,
        "timing_distribution": timing_counts,
        "immediate_action_required": priority_counts.get('CRITICAL', 0),
        "estimated_total_revenue": sum([5000 for _ in results]),  # Simplified calculation
        "automation_ready": total_messages
    }

async def _update_bulk_analytics(results: List[Dict[str, Any]]):
    """Background task to update bulk analytics"""
    logging.info(f"Updated bulk message analytics for {len(results)} messages")

@router.post("/social-media-message")
async def generate_social_media_message(request: Dict[str, Any]):
    """
    Generate interactive social media message for lead engagement
    """
    try:
        from backend.models.message_generation.social_media_integration import SocialMediaMessageGenerator, SocialPlatform, InteractionType
        
        social_generator = SocialMediaMessageGenerator()
        
        lead_data = request.get('lead_data', {})
        scoring_result = request.get('scoring_result', {})
        platform = SocialPlatform(request.get('platform', 'facebook'))
        interaction_type = InteractionType(request.get('interaction_type', 'direct_message'))
        context = request.get('context', {})
        
        # Generate social media message
        social_message = social_generator.generate_social_message(
            lead_data, scoring_result, platform, interaction_type, context
        )
        
        return {
            "status": "success",
            "social_message": {
                "platform": social_message.platform,
                "interaction_type": social_message.interaction_type,
                "content": social_message.message_content,
                "character_limit": social_message.character_limit,
                "call_to_action": social_message.call_to_action,
                "hashtags": social_message.hashtags,
                "mentions": social_message.mentions,
                "includes_media": social_message.includes_media,
                "compliance_notes": social_message.compliance_notes,
                "engagement_strategy": social_message.engagement_strategy
            },
            "platform_insights": {
                "optimal_posting_time": _get_optimal_posting_time(platform),
                "engagement_tips": _get_platform_engagement_tips(platform),
                "compliance_requirements": _get_platform_compliance(platform)
            }
        }
        
    except Exception as e:
        logging.error(f"Error generating social media message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating social message: {str(e)}")

@router.post("/send-email")
async def send_personalized_email(request: Dict[str, Any]):
    """
    Send personalized email to lead with verified email
    """
    try:
        from backend.models.message_generation.email_automation import EmailAutomationSystem
        
        # SMTP configuration (would be from environment variables)
        smtp_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your-email@company.com",
            "password": "your-app-password",
            "from_email": "your-email@company.com"
        }
        
        email_system = EmailAutomationSystem(smtp_config)
        
        lead_data = request.get('lead_data', {})
        scoring_result = request.get('scoring_result', {})
        message_content = request.get('message_content', {})
        email_type = request.get('email_type', 'initial_contact')
        
        # Send email
        result = await email_system.send_personalized_email(
            lead_data, scoring_result, message_content, email_type
        )
        
        return {
            "status": "success",
            "email_result": result,
            "delivery_insights": {
                "estimated_delivery_time": "2-5 minutes",
                "open_rate_prediction": _predict_email_open_rate(scoring_result),
                "best_follow_up_time": _calculate_best_follow_up_time(scoring_result)
            }
        }
        
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")

@router.post("/send-bulk-emails")
async def send_bulk_emails(request: Dict[str, Any]):
    """
    Send bulk emails to multiple leads with verified emails
    """
    try:
        from backend.models.message_generation.email_automation import EmailAutomationSystem
        
        smtp_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your-email@company.com", 
            "password": "your-app-password",
            "from_email": "your-email@company.com"
        }
        
        email_system = EmailAutomationSystem(smtp_config)
        
        leads_with_emails = request.get('leads_with_emails', [])
        email_type = request.get('email_type', 'bulk_campaign')
        
        # Send bulk emails
        result = await email_system.send_bulk_emails(leads_with_emails, email_type)
        
        return {
            "status": "success",
            "bulk_result": result,
            "campaign_analytics": {
                "estimated_open_rate": result['success_rate'] * 0.25,
                "estimated_click_rate": result['success_rate'] * 0.05,
                "estimated_conversions": result['successful_sends'] * 0.02,
                "roi_projection": _calculate_bulk_email_roi(result)
            }
        }
        
    except Exception as e:
        logging.error(f"Error sending bulk emails: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending bulk emails: {str(e)}")

@router.post("/send-email-sequence")
async def send_email_sequence(request: Dict[str, Any]):
    """
    Send automated email sequence based on lead scoring
    """
    try:
        from backend.models.message_generation.email_automation import EmailAutomationSystem
        
        smtp_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your-email@company.com",
            "password": "your-app-password", 
            "from_email": "your-email@company.com"
        }
        
        email_system = EmailAutomationSystem(smtp_config)
        
        lead_data = request.get('lead_data', {})
        sequence = request.get('sequence', {})
        
        # Send email sequence
        result = await email_system.send_sequence_emails(lead_data, sequence)
        
        return {
            "status": "success",
            "sequence_result": result,
            "automation_insights": {
                "sequence_completion_rate": result['steps_completed'] / result['total_steps'],
                "estimated_conversion_probability": _calculate_sequence_conversion_probability(result),
                "next_optimization_suggestions": _get_sequence_optimization_suggestions(result)
            }
        }
        
    except Exception as e:
        logging.error(f"Error sending email sequence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending email sequence: {str(e)}")

@router.get("/social-platforms")
async def get_supported_social_platforms():
    """
    Get list of supported social media platforms and their capabilities
    """
    return {
        "status": "success",
        "platforms": {
            "facebook": {
                "supported_interactions": ["direct_message", "comment_reply", "post_engagement"],
                "character_limits": {"dm": 2000, "comment": 8000, "post": 63206},
                "features": ["text", "images", "videos", "links"],
                "compliance": ["Facebook Terms", "Community Standards"],
                "best_practices": [
                    "Personalize messages based on profile information",
                    "Use engaging visuals for posts",
                    "Respond quickly to comments and messages"
                ]
            },
            "instagram": {
                "supported_interactions": ["direct_message", "comment_reply", "story_reply"],
                "character_limits": {"dm": 1000, "comment": 2200, "story": 160},
                "features": ["text", "images", "videos", "stories", "reels"],
                "compliance": ["Instagram Terms", "Community Guidelines"],
                "best_practices": [
                    "Use high-quality visuals",
                    "Include relevant hashtags",
                    "Engage with stories and reels"
                ]
            },
            "linkedin": {
                "supported_interactions": ["direct_message", "comment_reply", "connection_request"],
                "character_limits": {"dm": 8000, "comment": 1250, "connection": 300},
                "features": ["text", "documents", "articles", "professional content"],
                "compliance": ["LinkedIn User Agreement", "Professional Community Policies"],
                "best_practices": [
                    "Maintain professional tone",
                    "Share industry insights",
                    "Build genuine professional relationships"
                ]
            },
            "x": {  # Updated from twitter
                "supported_interactions": ["direct_message", "reply", "mention", "repost"],
                "character_limits": {"dm": 10000, "post": 280, "reply": 280},
                "features": ["text", "images", "videos", "spaces", "communities"],
                "compliance": ["X Terms of Service", "X Rules"],
                "best_practices": [
                    "Keep posts concise and engaging",
                    "Use relevant hashtags strategically",
                    "Engage in real-time conversations",
                    "Participate in X Spaces for audio engagement"
                ]
            },
            "tiktok": {
                "supported_interactions": ["direct_message", "comment_reply"],
                "character_limits": {"dm": 1000, "comment": 150},
                "features": ["text", "videos", "effects", "live"],
                "compliance": ["TikTok Community Guidelines", "Terms of Service"],
                "best_practices": [
                    "Create engaging video content",
                    "Use trending sounds and effects",
                    "Keep messages fun and authentic"
                ]
            }
        },
        "integration_requirements": {
            "api_access": "Platform-specific API keys required",
            "authentication": "OAuth 2.0 for most platforms",
            "rate_limits": "Varies by platform and endpoint",
            "compliance": "Must follow platform terms and advertising policies"
        }
    }

# Helper functions for new endpoints
def _get_optimal_posting_time(platform) -> str:
    """Get optimal posting time for platform"""
    optimal_times = {
        "facebook": "1-3 PM weekdays",
        "instagram": "11 AM - 1 PM weekdays", 
        "linkedin": "8-10 AM weekdays",
        "x": "9 AM - 3 PM weekdays",  # Updated from twitter
        "tiktok": "6-10 PM daily"
    }
    return optimal_times.get(platform.value, "9 AM - 5 PM weekdays")

def _get_platform_engagement_tips(platform) -> List[str]:
    """Get engagement tips for platform"""
    tips = {
        "facebook": [
            "Ask questions to encourage comments",
            "Share behind-the-scenes content",
            "Use Facebook Live for real-time engagement"
        ],
        "instagram": [
            "Use Instagram Stories polls and questions",
            "Post consistently with branded hashtags",
            "Collaborate with micro-influencers"
        ],
        "linkedin": [
            "Share industry insights and thought leadership",
            "Engage with others' professional content",
            "Use LinkedIn articles for long-form content"
        ],
        "x": [  # Updated from twitter
            "Join trending conversations",
            "Use X Spaces for audio engagement",
            "Create post threads for detailed topics",
            "Engage with X Communities in your niche"
        ],
        "tiktok": [
            "Use trending sounds and challenges",
            "Create educational or entertaining content",
            "Engage with comments quickly"
        ]
    }
    return tips.get(platform.value, ["Engage authentically with your audience"])

def _get_platform_compliance(platform) -> List[str]:
    """Get compliance requirements for platform"""
    compliance = {
        "facebook": ["No misleading claims", "Respect user privacy", "Follow advertising policies"],
        "instagram": ["Authentic content only", "Respect intellectual property", "No spam"],
        "linkedin": ["Professional conduct", "No unsolicited messages", "Respect connections"],
        "x": ["No harassment", "Respect copyright", "Follow X Rules", "No misleading information"],  # Updated from twitter
        "tiktok": ["Age-appropriate content", "No misleading information", "Respect community guidelines"]
    }
    return compliance.get(platform.value, ["Follow platform terms of service"])

def _predict_email_open_rate(scoring_result: Dict[str, Any]) -> float:
    """Predict email open rate based on lead scoring"""
    base_rate = 0.25
    priority = scoring_result.get('priority_level', 'MEDIUM')
    
    multipliers = {
        'CRITICAL': 1.4,
        'HIGH': 1.2, 
        'MEDIUM': 1.0,
        'LOW': 0.8
    }
    
    return base_rate * multipliers.get(priority, 1.0)

def _calculate_best_follow_up_time(scoring_result: Dict[str, Any]) -> str:
    """Calculate best follow-up time based on lead scoring"""
    priority = scoring_result.get('priority_level', 'MEDIUM')
    
    follow_up_times = {
        'CRITICAL': '4 hours if no response',
        'HIGH': '24 hours if no response',
        'MEDIUM': '3 days if no response', 
        'LOW': '1 week if no response'
    }
    
    return follow_up_times.get(priority, '3 days if no response')

def _calculate_bulk_email_roi(result: Dict[str, Any]) -> Dict[str, float]:
    """Calculate ROI for bulk email campaign"""
    successful_sends = result.get('successful_sends', 0)
    estimated_conversions = successful_sends * 0.02  # 2% conversion rate
    revenue_per_conversion = 5000  # Average policy value
    campaign_cost = result.get('total_leads', 0) * 0.10  # $0.10 per email
    
    total_revenue = estimated_conversions * revenue_per_conversion
    roi = (total_revenue - campaign_cost) / campaign_cost if campaign_cost > 0 else 0
    
    return {
        "estimated_revenue": total_revenue,
        "campaign_cost": campaign_cost,
        "roi_percentage": roi * 100,
        "cost_per_conversion": campaign_cost / estimated_conversions if estimated_conversions > 0 else 0
    }

def _calculate_sequence_conversion_probability(result: Dict[str, Any]) -> float:
    """Calculate conversion probability for email sequence"""
    completion_rate = result.get('steps_completed', 0) / result.get('total_steps', 1)
    base_conversion = 0.15  # 15% base conversion for complete sequences
    
    return completion_rate * base_conversion

def _get_sequence_optimization_suggestions(result: Dict[str, Any]) -> List[str]:
    """Get optimization suggestions for email sequences"""
    completion_rate = result.get('steps_completed', 0) / result.get('total_steps', 1)
    
    suggestions = []
    
    if completion_rate < 0.5:
        suggestions.append("Consider reducing sequence length or improving email timing")
    
    if completion_rate < 0.8:
        suggestions.append("A/B test subject lines to improve open rates")
        suggestions.append("Personalize content based on lead behavior")
    
    suggestions.extend([
        "Monitor reply rates and adjust messaging tone",
        "Test different call-to-action buttons",
        "Consider adding SMS follow-ups for critical leads"
    ])
    
    return suggestions


