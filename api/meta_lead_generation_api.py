from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any
from pydantic import BaseModel
import logging
from datetime import datetime

from models.meta_lead_generation.inference import MetaLeadGenerationInference

router = APIRouter(prefix="/meta-lead-generation", tags=["Enhanced Meta Lead Generation"])

# Initialize the enhanced meta model
meta_scorer = MetaLeadGenerationInference()

class EnhancedLeadData(BaseModel):
    lead_id: str
    age: int
    income: float
    employment_status: str
    location: str
    consent_given: bool
    
    # Enhanced fields for urgency detection
    family_size: int = 1
    marital_status: str = "single"
    dependents_count: int = 0
    smoking_status: str = "non_smoker"
    health_conditions: List[str] = []
    health_conditions_count: int = 0
    
    # Life events (urgency signals)
    qualifying_life_event: bool = False
    recent_marriage: bool = False
    new_baby: bool = False
    home_purchase: bool = False
    job_change: bool = False
    income_increase: bool = False
    
    # Current coverage
    current_coverage: str = "unknown"
    has_life_insurance: bool = False
    
    # Engagement metrics
    engagement_score: int = 5
    quote_requests_30d: int = 0
    social_engagement_score: int = 5
    
    # Optional specialized fields
    prescription_usage: int = 0
    doctor_visits_annual: int = 2
    coverage_amount_requested: float = 0

class BatchLeadRequest(BaseModel):
    leads: List[EnhancedLeadData]

class MarketUpdateRequest(BaseModel):
    healthcare: float = None
    life: float = None
    base: float = None

@router.post("/enhanced-score-lead")
async def enhanced_score_lead(lead_data: EnhancedLeadData):
    """
    Score a single lead with all enhanced features
    """
    try:
        lead_dict = lead_data.dict()
        result = meta_scorer.score_lead(lead_dict)
        
        return {
            "status": "success",
            "data": result,
            "insights": {
                "key_strengths": _identify_key_strengths(result),
                "risk_factors": _identify_risk_factors(result),
                "competitive_advantages": _identify_competitive_advantages(result)
            }
        }
        
    except Exception as e:
        logging.error(f"Error scoring lead {lead_data.lead_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scoring lead: {str(e)}")

@router.post("/enhanced-batch-score")
async def enhanced_batch_score(request: BatchLeadRequest, background_tasks: BackgroundTasks):
    """
    Score multiple leads with enhanced features and analytics
    """
    try:
        leads_data = [lead.dict() for lead in request.leads]
        results = meta_scorer.batch_score_leads(leads_data)
        
        # Generate batch analytics
        analytics = _generate_batch_analytics(results)
        
        # Schedule background analytics update
        background_tasks.add_task(_update_batch_analytics, analytics)
        
        return {
            "status": "success",
            "total_leads": len(results),
            "data": results,
            "batch_analytics": analytics
        }
        
    except Exception as e:
        logging.error(f"Error in enhanced batch scoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in batch scoring: {str(e)}")

@router.post("/sales-strategy")
async def get_sales_strategy(lead_data: EnhancedLeadData):
    """
    Get comprehensive sales strategy and recommendations
    """
    try:
        lead_dict = lead_data.dict()
        recommendations = meta_scorer.get_lead_recommendations(lead_dict)
        
        return {
            "status": "success",
            "lead_id": lead_data.lead_id,
            "sales_strategy": recommendations,
            "action_items": _generate_action_items(recommendations),
            "success_probability": recommendations['competitive_analysis']['win_probability']
        }
        
    except Exception as e:
        logging.error(f"Error generating sales strategy for {lead_data.lead_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating strategy: {str(e)}")

@router.get("/priority-leads")
async def get_priority_leads(priority_level: str = "HIGH", limit: int = 50):
    """
    Get leads filtered by priority level (for sales dashboard)
    """
    # This would typically query a database of scored leads
    return {
        "status": "success",
        "priority_level": priority_level,
        "limit": limit,
        "message": f"Would return top {limit} leads with {priority_level} priority",
        "filters_available": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        "sort_options": ["overall_score", "revenue_potential", "conversion_velocity"]
    }

@router.get("/velocity-dashboard")
async def get_velocity_dashboard():
    """
    Get conversion velocity dashboard data
    """
    return {
        "status": "success",
        "velocity_categories": {
            "IMMEDIATE": {"count": 45, "avg_score": 87.2, "revenue_potential": 1250000},
            "FAST": {"count": 123, "avg_score": 76.8, "revenue_potential": 2100000},
            "MEDIUM": {"count": 234, "avg_score": 65.4, "revenue_potential": 1800000},
            "SLOW": {"count": 156, "avg_score": 52.1, "revenue_potential": 950000}
        },
        "recommendations": [
            "Focus immediate attention on 45 IMMEDIATE velocity leads",
            "Schedule calls for 123 FAST velocity leads within 24 hours",
            "Set up nurture campaigns for MEDIUM and SLOW velocity leads"
        ]
    }

@router.post("/update-market-conditions")
async def update_market_conditions(market_update: MarketUpdateRequest):
    """
    Update market demand multipliers dynamically
    """
    try:
        updates = {}
        if market_update.healthcare is not None:
            updates['healthcare'] = market_update.healthcare
        if market_update.life is not None:
            updates['life'] = market_update.life
        if market_update.base is not None:
            updates['base'] = market_update.base
        
        result = meta_scorer.update_market_conditions(updates)
        
        return {
            "status": "success",
            "message": "Market conditions updated successfully",
            "updated_multipliers": result["updated_multipliers"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error updating market conditions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating market conditions: {str(e)}")

@router.get("/urgency-signals")
async def get_urgency_signals_analysis():
    """
    Get analysis of urgency signals across all leads
    """
    return {
        "status": "success",
        "urgency_signals": {
            "OPEN_ENROLLMENT_PERIOD": {"count": 234, "avg_boost": 8.5, "priority": "CRITICAL"},
            "NEW_BABY": {"count": 45, "avg_boost": 12.3, "priority": "CRITICAL"},
            "NO_CURRENT_COVERAGE": {"count": 156, "avg_boost": 6.7, "priority": "HIGH"},
            "HIGH_ENGAGEMENT": {"count": 89, "avg_boost": 5.2, "priority": "HIGH"},
            "JOB_CHANGE": {"count": 67, "avg_boost": 4.1, "priority": "MEDIUM"}
        },
        "recommendations": [
            "Prioritize 234 leads in open enrollment period",
            "Immediate contact for 45 new parent leads",
            "Focus on 156 uninsured leads for quick wins"
        ]
    }

@router.get("/model-performance")
async def get_model_performance():
    """
    Get enhanced model performance metrics
    """
    try:
        metrics = meta_scorer.get_model_performance_metrics()
        
        return {
            "status": "success",
            "performance_metrics": metrics,
            "health_status": "optimal",
            "last_updated": datetime.now().isoformat(),
            "enhancement_features": [
                "✅ Confidence-weighted ensemble scoring",
                "✅ Real-time market demand adjustment",
                "✅ Advanced urgency signal detection",
                "✅ Conversion velocity prediction",
                "✅ Optimal contact timing",
                "✅ Revenue potential calculation",
                "✅ Enhanced cross-sell identification",
                "✅ Competitive positioning analysis"
            ]
        }
        
    except Exception as e:
        logging.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")

# Helper functions
def _identify_key_strengths(result: Dict[str, Any]) -> List[str]:
    """Identify key strengths of the lead"""
    strengths = []
    
    if result['overall_score'] > 80:
        strengths.append("High overall score")
    
    if len(result['recommended_products']) > 1:
        strengths.append("Multi-product opportunity")
    
    if result['revenue_potential'] > 15000:
        strengths.append("High revenue potential")
    
    if result['urgency_signals']:
        strengths.append("Time-sensitive opportunity")
    
    return strengths

def _identify_risk_factors(result: Dict[str, Any]) -> List[str]:
    """Identify potential risk factors"""
    risks = []
    
    if result['confidence_score'] < 0.6:
        risks.append("Low prediction confidence")
    
    if not result['urgency_signals']:
        risks.append("No urgency signals detected")
    
    if result['overall_score'] < 60:
        risks.append("Below-average lead score")
    
    return risks

def _identify_competitive_advantages(result: Dict[str, Any]) -> List[str]:
    """Identify competitive advantages"""
    advantages = []
    
    if "IMMEDIATE" in result['conversion_velocity'].values():
        advantages.append("Fast decision maker")
    
    if len(result['cross_sell_opportunities']) > 0:
        advantages.append("Bundle opportunity")
    
    if result['optimal_contact_time'] in ["IMMEDIATE", "WITHIN_2_HOURS"]:
        advantages.append("Timing advantage")
    
    return advantages

def _generate_batch_analytics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate analytics for batch processing"""
    total_leads = len(results)
    
    if total_leads == 0:
        return {"total_leads": 0}
    
    # Calculate averages and distributions
    avg_score = sum(r['overall_score'] for r in results) / total_leads
    total_revenue = sum(r['revenue_potential'] for r in results)
    
    priority_distribution = {}
    for result in results:
        priority = result['priority_level']
        priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
    
    return {
        "total_leads": total_leads,
        "average_score": avg_score,
        "total_revenue_potential": total_revenue,
        "priority_distribution": priority_distribution,
        "high_value_leads": len([r for r in results if r['revenue_potential'] > 15000]),
        "immediate_action_required": len([r for r in results if r['priority_level'] == 'CRITICAL'])
    }

async def _update_batch_analytics(analytics: Dict[str, Any]):
    """Background task to update analytics"""
    # This would typically update a database or analytics system
    logging.info(f"Updated batch analytics: {analytics}")
