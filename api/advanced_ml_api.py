"""
Advanced Machine Learning API

REST API endpoints for deep learning models, ensemble optimization,
behavioral prediction, and market trend analysis.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime

from advanced_ml.deep_learning_models import deep_learning_manager
from advanced_ml.ensemble_optimization import ensemble_optimizer
from advanced_ml.behavioral_prediction import behavioral_prediction_engine
from advanced_ml.market_trend_analysis import market_trend_engine

router = APIRouter(prefix="/advanced-ml", tags=["Advanced Machine Learning"])
logger = logging.getLogger(__name__)

# Request Models
class DeepLearningRequest(BaseModel):
    lead_features: Dict[str, Any]
    model_type: str = "deep_scorer"  # deep_scorer, behavioral_sequence, autoencoder

class EnsembleOptimizationRequest(BaseModel):
    training_data: List[Dict[str, Any]]
    target_column: str = "conversion_score"
    optimization_method: str = "optuna"  # optuna, grid_search, random_search

class BehavioralPredictionRequest(BaseModel):
    lead_data: Dict[str, Any]
    include_insights: bool = True

class MarketTrendRequest(BaseModel):
    lead_score: float
    include_market_data: bool = True

class BatchScoringRequest(BaseModel):
    leads: List[Dict[str, Any]]
    use_ensemble: bool = True
    use_market_trends: bool = True

# Deep Learning Endpoints
@router.post("/deep-learning/predict")
async def predict_with_deep_learning(request: DeepLearningRequest):
    """
    Make predictions using deep learning models
    """
    try:
        if request.model_type == "autoencoder":
            # Anomaly detection
            result = await deep_learning_manager.detect_anomalies(request.lead_features)
        else:
            # Lead scoring
            result = await deep_learning_manager.predict_lead_score(
                request.lead_features, 
                request.model_type
            )
        
        return {
            "status": "success",
            "prediction": result,
            "model_type": request.model_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Deep learning prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deep-learning/train")
async def train_deep_learning_model(
    training_data: List[Dict[str, Any]],
    model_type: str = "deep_scorer",
    background_tasks: BackgroundTasks = None
):
    """
    Train deep learning models
    """
    try:
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        if background_tasks:
            # Train in background
            if model_type == "deep_scorer":
                background_tasks.add_task(
                    deep_learning_manager.train_deep_lead_scorer, df
                )
            elif model_type == "behavioral_sequence":
                background_tasks.add_task(
                    deep_learning_manager.train_behavioral_sequence_model, df
                )
            elif model_type == "autoencoder":
                background_tasks.add_task(
                    deep_learning_manager.train_autoencoder_features, df
                )
            
            return {
                "status": "training_started",
                "message": f"Training {model_type} model in background",
                "estimated_time": "10-30 minutes"
            }
        else:
            # Train synchronously
            if model_type == "deep_scorer":
                performance = await deep_learning_manager.train_deep_lead_scorer(df)
            elif model_type == "behavioral_sequence":
                performance = await deep_learning_manager.train_behavioral_sequence_model(df)
            elif model_type == "autoencoder":
                performance = await deep_learning_manager.train_autoencoder_features(df)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return {
                "status": "training_completed",
                "performance": performance.__dict__,
                "model_type": model_type
            }
        
    except Exception as e:
        logger.error(f"Deep learning training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Ensemble Optimization Endpoints
@router.post("/ensemble/optimize")
async def optimize_ensemble(request: EnsembleOptimizationRequest):
    """
    Optimize ensemble models with hyperparameter tuning
    """
    try:
        import pandas as pd
        
        # Convert training data
        df = pd.DataFrame(request.training_data)
        X = df.drop(columns=[request.target_column])
        y = df[request.target_column]
        
        # Optimize ensemble
        performance = await ensemble_optimizer.optimize_ensemble(
            X, y, optimization_method=request.optimization_method
        )
        
        return {
            "status": "optimization_completed",
            "performance": performance.__dict__,
            "optimization_method": request.optimization_method,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ensemble optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ensemble/predict")
async def predict_with_ensemble(
    lead_features: Dict[str, Any],
    method: str = "stacking"  # voting, stacking
):
    """
    Make predictions using optimized ensemble
    """
    try:
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame([lead_features])
        
        # Make prediction
        prediction = await ensemble_optimizer.predict_with_ensemble(df, method)
        
        # Get feature importance
        feature_importance = await ensemble_optimizer.get_feature_importance()
        
        return {
            "status": "success",
            "prediction": float(prediction[0]),
            "ensemble_method": method,
            "feature_importance": feature_importance,
            "model_count": len(ensemble_optimizer.trained_models),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ensemble/update-weights")
async def update_ensemble_weights(
    recent_data: List[Dict[str, Any]],
    target_column: str = "conversion_score",
    method: str = "performance_based"
):
    """
    Update ensemble weights based on recent performance
    """
    try:
        import pandas as pd
        
        df = pd.DataFrame(recent_data)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        await ensemble_optimizer.update_ensemble_weights(X, y, method)
        
        return {
            "status": "weights_updated",
            "new_weights": ensemble_optimizer.ensemble_weights,
            "update_method": method,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Weight update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Behavioral Prediction Endpoints
@router.post("/behavioral/predict")
async def predict_behavioral_insights(request: BehavioralPredictionRequest):
    """
    Generate behavioral predictions and insights
    """
    try:
        insights = await behavioral_prediction_engine.predict_behavioral_insights(
            request.lead_data
        )
        
        response = {
            "status": "success",
            "insights": {
                "lead_id": insights.lead_id,
                "next_best_action": insights.next_best_action.value,
                "action_confidence": insights.action_confidence,
                "churn_risk": insights.churn_risk.value,
                "churn_probability": insights.churn_probability,
                "engagement_pattern": insights.engagement_pattern.value,
                "engagement_score": insights.engagement_score,
                "predicted_conversion_timeline": insights.predicted_conversion_timeline,
                "prediction_timestamp": insights.prediction_timestamp.isoformat()
            }
        }
        
        if request.include_insights:
            response["insights"].update({
                "recommended_touchpoints": insights.recommended_touchpoints,
                "behavioral_triggers": insights.behavioral_triggers,
                "risk_factors": insights.risk_factors,
                "opportunity_indicators": insights.opportunity_indicators
            })
        
        return response
        
    except Exception as e:
        logger.error(f"Behavioral prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/behavioral/train")
async def train_behavioral_models(
    training_data: List[Dict[str, Any]],
    background_tasks: BackgroundTasks = None
):
    """
    Train behavioral prediction models
    """
    try:
        import pandas as pd
        
        df = pd.DataFrame(training_data)
        
        if background_tasks:
            background_tasks.add_task(
                behavioral_prediction_engine.train_behavioral_models, df
            )
            return {
                "status": "training_started",
                "message": "Training behavioral models in background"
            }
        else:
            results = await behavioral_prediction_engine.train_behavioral_models(df)
            return {
                "status": "training_completed",
                "model_performance": results,
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Behavioral training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Market Trend Analysis Endpoints
@router.get("/market-trends/collect-data")
async def collect_market_data():
    """
    Collect current market data and indicators
    """
    try:
        indicators = await market_trend_engine.collect_market_data()
        
        return {
            "status": "success",
            "market_indicators": {
                "economic": {
                    "gdp_growth": indicators.gdp_growth,
                    "unemployment_rate": indicators.unemployment_rate,
                    "inflation_rate": indicators.inflation_rate,
                    "interest_rates": indicators.interest_rates,
                    "consumer_confidence": indicators.consumer_confidence
                },
                "market": {
                    "stock_market_performance": indicators.stock_market_performance,
                    "insurance_sector_performance": indicators.insurance_sector_performance,
                    "volatility_index": indicators.volatility_index
                },
                "industry": {
                    "insurance_penetration": indicators.insurance_penetration,
                    "digital_adoption_rate": indicators.digital_adoption_rate,
                    "competitor_activity": indicators.competitor_activity
                },
                "seasonal": {
                    "quarter": indicators.quarter,
                    "is_holiday_season": indicators.is_holiday_season,
                    "is_tax_season": indicators.is_tax_season
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market data collection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/market-trends/analyze")
async def analyze_market_trends(include_market_data: bool = True):
    """
    Analyze current market trends and conditions
    """
    try:
        # Collect fresh data if requested
        if include_market_data:
            indicators = await market_trend_engine.collect_market_data()
        else:
            # Use most recent data
            latest_date = max(market_trend_engine.market_data.keys()) if market_trend_engine.market_data else datetime.now().date()
            indicators = market_trend_engine.market_data.get(latest_date)
            
            if not indicators:
                indicators = await market_trend_engine.collect_market_data()
        
        # Analyze trends
        analysis = await market_trend_engine.analyze_market_trends(indicators)
        
        return {
            "status": "success",
            "trend_analysis": {
                "market_condition": analysis.market_condition.value,
                "seasonal_pattern": analysis.seasonal_pattern.value,
                "trend_direction": analysis.trend_direction.value,
                "trend_strength": analysis.trend_strength,
                "volatility_score": analysis.volatility_score,
                "opportunity_score": analysis.opportunity_score,
                "risk_score": analysis.risk_score,
                "confidence_level": analysis.confidence_level,
                "recommended_adjustments": analysis.recommended_adjustments,
                "analysis_timestamp": analysis.analysis_timestamp.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Market trend analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/market-trends/adapt-scoring")
async def adapt_lead_scoring(request: MarketTrendRequest):
    """
    Adapt lead scoring based on market trends
    """
    try:
        # Get or collect market data
        if request.include_market_data:
            indicators = await market_trend_engine.collect_market_data()
        else:
            latest_date = max(market_trend_engine.market_data.keys()) if market_trend_engine.market_data else datetime.now().date()
            indicators = market_trend_engine.market_data.get(latest_date)
            
            if not indicators:
                indicators = await market_trend_engine.collect_market_data()
        
        # Analyze trends
        trend_analysis = await market_trend_engine.analyze_market_trends(indicators)
        
        # Adapt scoring
        adapted_scoring = await market_trend_engine.adapt_lead_scoring(
            request.lead_score, trend_analysis
        )
        
        return {
            "status": "success",
            "adapted_scoring": adapted_scoring,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Scoring adaptation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comprehensive Batch Processing
@router.post("/batch-scoring")
async def comprehensive_batch_scoring(request: BatchScoringRequest):
    """
    Comprehensive batch scoring with all advanced ML features
    """
    try:
        results = []
        
        # Collect market data once for all leads
        if request.use_market_trends:
            indicators = await market_trend_engine.collect_market_data()
            trend_analysis = await market_trend_engine.analyze_market_trends(indicators)
        
        for lead_data in request.leads:
            lead_result = {
                "lead_id": lead_data.get("lead_id", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                # Base scoring
                if request.use_ensemble:
                    import pandas as pd
                    df = pd.DataFrame([lead_data])
                    base_score = await ensemble_optimizer.predict_with_ensemble(df)
                    lead_result["base_score"] = float(base_score[0])
                else:
                    # Use deep learning
                    dl_result = await deep_learning_manager.predict_lead_score(lead_data)
                    lead_result["base_score"] = dl_result.get("lead_score", 50.0)
                
                # Behavioral insights
                behavioral_insights = await behavioral_prediction_engine.predict_behavioral_insights(lead_data)
                lead_result["behavioral_insights"] = {
                    "next_best_action": behavioral_insights.next_best_action.value,
                    "churn_risk": behavioral_insights.churn_risk.value,
                    "engagement_pattern": behavioral_insights.engagement_pattern.value,
                    "predicted_timeline": behavioral_insights.predicted_conversion_timeline
                }
                
                # Market trend adaptation
                if request.use_market_trends:
                    adapted_scoring = await market_trend_engine.adapt_lead_scoring(
                        lead_result["base_score"], trend_analysis
                    )
                    lead_result["market_adapted_score"] = adapted_scoring["adjusted_score"]
                    lead_result["market_adjustments"] = adapted_scoring
                else:
                    lead_result["market_adapted_score"] = lead_result["base_score"]
                
                lead_result["status"] = "success"
                
            except Exception as e:
                logger.error(f"Error processing lead {lead_data.get('lead_id')}: {e}")
                lead_result["status"] = "error"
                lead_result["error"] = str(e)
                lead_result["base_score"] = 50.0  # Default score
                lead_result["market_adapted_score"] = 50.0
            
            results.append(lead_result)
        
        # Summary statistics
        successful_results = [r for r in results if r["status"] == "success"]
        summary = {
            "total_leads": len(request.leads),
            "successful_predictions": len(successful_results),
            "average_base_score": sum(r["base_score"] for r in successful_results) / len(successful_results) if successful_results else 0,
            "average_adapted_score": sum(r["market_adapted_score"] for r in successful_results) / len(successful_results) if successful_results else 0,
            "processing_time": datetime.now().isoformat()
        }
        
        return {
            "status": "completed",
            "results": results,
            "summary": summary,
            "market_condition": trend_analysis.market_condition.value if request.use_market_trends else "not_analyzed"
        }
        
    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Performance and Monitoring
@router.get("/performance/summary")
async def get_performance_summary():
    """
    Get performance summary of all advanced ML models
    """
    try:
        summary = {
            "deep_learning": {
                "models_trained": len(deep_learning_manager.models),
                "performance_history": deep_learning_manager.performance_history
            },
            "ensemble": {
                "active_models": len(ensemble_optimizer.trained_models),
                "ensemble_weights": ensemble_optimizer.ensemble_weights,
                "performance_history": [p.__dict__ for p in ensemble_optimizer.performance_history]
            },
            "behavioral": {
                "model_performance": behavioral_prediction_engine.model_performance
            },
            "market_trends": {
                "data_points": len(market_trend_engine.market_data),
                "last_update": max(market_trend_engine.market_data.keys()).isoformat() if market_trend_engine.market_data else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "performance_summary": summary
        }
        
    except Exception as e:
        logger.error(f"Performance summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/continuous-improvement")
async def trigger_continuous_improvement(
    new_data: List[Dict[str, Any]],
    target_column: str = "conversion_score",
    background_tasks: BackgroundTasks = None
):
    """
    Trigger continuous improvement for all models
    """
    try:
        import pandas as pd
        
        df = pd.DataFrame(new_data)
        
        if background_tasks:
            # Run improvements in background
            background_tasks.add_task(
                ensemble_optimizer.continuous_improvement, df, target_column
            )
            background_tasks.add_task(
                behavioral_prediction_engine.train_behavioral_models, df
            )
            
            return {
                "status": "improvement_started",
                "message": "Continuous improvement running in background",
                "data_points": len(new_data)
            }
        else:
            # Run synchronously
            await ensemble_optimizer.continuous_improvement(df, target_column)
            await behavioral_prediction_engine.train_behavioral_models(df)
            
            return {
                "status": "improvement_completed",
                "message": "All models updated with new data",
                "data_points": len(new_data),
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Continuous improvement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))