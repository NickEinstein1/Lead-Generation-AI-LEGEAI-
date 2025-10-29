"""
Revenue Attribution Engine

Advanced revenue attribution system that tracks and analyzes revenue sources,
customer lifetime value, and attribution across multiple touchpoints.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json

logger = logging.getLogger(__name__)

class AttributionModel(Enum):
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"
    DATA_DRIVEN = "data_driven"

@dataclass
class TouchPoint:
    """Individual customer touchpoint"""
    touchpoint_id: str
    customer_id: str
    source: str
    medium: str
    campaign: str
    content: str
    timestamp: datetime
    value: float = 0.0
    conversion_value: float = 0.0
    attribution_weight: float = 0.0

@dataclass
class CustomerJourney:
    """Complete customer journey with all touchpoints"""
    customer_id: str
    first_touch: datetime
    last_touch: datetime
    total_touchpoints: int
    touchpoints: List[TouchPoint] = field(default_factory=list)
    total_revenue: float = 0.0
    conversion_date: Optional[datetime] = None
    journey_duration_days: int = 0

@dataclass
class AttributionResult:
    """Attribution analysis result"""
    source: str
    medium: str
    campaign: str
    attributed_revenue: float
    attributed_conversions: int
    cost: float = 0.0
    roi: float = 0.0
    cpa: float = 0.0  # Cost per acquisition
    roas: float = 0.0  # Return on ad spend

class RevenueAttributionEngine:
    """Advanced revenue attribution engine"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        
        # Attribution model weights
        self.attribution_weights = {
            AttributionModel.FIRST_TOUCH: {"first": 1.0},
            AttributionModel.LAST_TOUCH: {"last": 1.0},
            AttributionModel.LINEAR: {"equal": True},
            AttributionModel.TIME_DECAY: {"decay_rate": 0.7},
            AttributionModel.POSITION_BASED: {"first": 0.4, "last": 0.4, "middle": 0.2},
        }
        
        # Data storage keys
        self.storage_keys = {
            'touchpoints': 'attribution:touchpoints',
            'journeys': 'attribution:journeys',
            'conversions': 'attribution:conversions',
            'costs': 'attribution:costs'
        }
        
        logger.info("Revenue Attribution Engine initialized")
    
    async def track_touchpoint(self, touchpoint_data: Dict[str, Any]) -> str:
        """Track a customer touchpoint"""
        
        try:
            touchpoint = TouchPoint(
                touchpoint_id=touchpoint_data.get('touchpoint_id', self._generate_touchpoint_id()),
                customer_id=touchpoint_data['customer_id'],
                source=touchpoint_data.get('source', 'unknown'),
                medium=touchpoint_data.get('medium', 'unknown'),
                campaign=touchpoint_data.get('campaign', 'unknown'),
                content=touchpoint_data.get('content', ''),
                timestamp=datetime.fromisoformat(touchpoint_data.get('timestamp', datetime.utcnow().isoformat())),
                value=touchpoint_data.get('value', 0.0)
            )
            
            # Store touchpoint
            await self._store_touchpoint(touchpoint)
            
            # Update customer journey
            await self._update_customer_journey(touchpoint)
            
            logger.debug(f"Tracked touchpoint {touchpoint.touchpoint_id} for customer {touchpoint.customer_id}")
            
            return touchpoint.touchpoint_id
            
        except Exception as e:
            logger.error(f"Error tracking touchpoint: {e}")
            raise
    
    async def track_conversion(self, conversion_data: Dict[str, Any]) -> str:
        """Track a conversion event"""
        
        try:
            customer_id = conversion_data['customer_id']
            revenue = conversion_data.get('revenue', 0.0)
            conversion_date = datetime.fromisoformat(
                conversion_data.get('conversion_date', datetime.utcnow().isoformat())
            )
            
            # Get customer journey
            journey = await self._get_customer_journey(customer_id)
            
            if journey:
                # Update journey with conversion
                journey.total_revenue += revenue
                journey.conversion_date = conversion_date
                journey.journey_duration_days = (conversion_date - journey.first_touch).days
                
                # Store updated journey
                await self._store_customer_journey(journey)
                
                # Store conversion record
                conversion_record = {
                    'customer_id': customer_id,
                    'revenue': revenue,
                    'conversion_date': conversion_date.isoformat(),
                    'journey_id': f"journey_{customer_id}"
                }
                
                if self.redis_client:
                    self.redis_client.hset(
                        self.storage_keys['conversions'],
                        f"conversion_{customer_id}_{int(conversion_date.timestamp())}",
                        json.dumps(conversion_record)
                    )
                
                logger.info(f"Tracked conversion for customer {customer_id}: ${revenue}")
                
                return f"conversion_{customer_id}_{int(conversion_date.timestamp())}"
            else:
                logger.warning(f"No journey found for customer {customer_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error tracking conversion: {e}")
            raise
    
    async def calculate_attribution(self, 
                                  model: AttributionModel = AttributionModel.LINEAR,
                                  time_range_days: int = 30) -> List[AttributionResult]:
        """Calculate revenue attribution using specified model"""
        
        try:
            # Get conversions in time range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=time_range_days)
            
            conversions = await self._get_conversions_in_range(start_date, end_date)
            
            attribution_results = {}
            
            for conversion in conversions:
                customer_id = conversion['customer_id']
                revenue = conversion['revenue']
                
                # Get customer journey
                journey = await self._get_customer_journey(customer_id)
                
                if journey and journey.touchpoints:
                    # Calculate attribution weights for this journey
                    weights = self._calculate_attribution_weights(journey.touchpoints, model)
                    
                    # Distribute revenue based on weights
                    for touchpoint, weight in weights.items():
                        attributed_revenue = revenue * weight
                        
                        key = f"{touchpoint.source}|{touchpoint.medium}|{touchpoint.campaign}"
                        
                        if key not in attribution_results:
                            attribution_results[key] = {
                                'source': touchpoint.source,
                                'medium': touchpoint.medium,
                                'campaign': touchpoint.campaign,
                                'attributed_revenue': 0.0,
                                'attributed_conversions': 0,
                                'cost': 0.0
                            }
                        
                        attribution_results[key]['attributed_revenue'] += attributed_revenue
                        attribution_results[key]['attributed_conversions'] += weight
            
            # Get cost data and calculate ROI/ROAS
            results = []
            for key, data in attribution_results.items():
                cost = await self._get_channel_cost(data['source'], data['medium'], data['campaign'], time_range_days)
                
                result = AttributionResult(
                    source=data['source'],
                    medium=data['medium'],
                    campaign=data['campaign'],
                    attributed_revenue=data['attributed_revenue'],
                    attributed_conversions=int(data['attributed_conversions']),
                    cost=cost
                )
                
                # Calculate metrics
                if cost > 0:
                    result.roi = ((result.attributed_revenue - cost) / cost) * 100
                    result.roas = result.attributed_revenue / cost
                    result.cpa = cost / max(1, result.attributed_conversions)
                
                results.append(result)
            
            # Sort by attributed revenue
            results.sort(key=lambda x: x.attributed_revenue, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating attribution: {e}")
            raise
    
    async def get_customer_ltv_analysis(self, segment: str = None) -> Dict[str, Any]:
        """Analyze customer lifetime value by segments"""
        
        try:
            # Get all customer journeys
            journeys = await self._get_all_customer_journeys()
            
            if segment:
                # Filter by segment (this would be enhanced with actual segmentation logic)
                journeys = [j for j in journeys if self._customer_in_segment(j.customer_id, segment)]
            
            if not journeys:
                return {"error": "No customer data available"}
            
            # Calculate LTV metrics
            revenues = [j.total_revenue for j in journeys if j.total_revenue > 0]
            journey_durations = [j.journey_duration_days for j in journeys if j.journey_duration_days > 0]
            touchpoint_counts = [j.total_touchpoints for j in journeys]
            
            ltv_analysis = {
                'total_customers': len(journeys),
                'converted_customers': len(revenues),
                'conversion_rate': (len(revenues) / len(journeys)) * 100 if journeys else 0,
                'average_ltv': np.mean(revenues) if revenues else 0,
                'median_ltv': np.median(revenues) if revenues else 0,
                'ltv_std': np.std(revenues) if revenues else 0,
                'average_journey_days': np.mean(journey_durations) if journey_durations else 0,
                'average_touchpoints': np.mean(touchpoint_counts) if touchpoint_counts else 0,
                'ltv_percentiles': {
                    '25th': np.percentile(revenues, 25) if revenues else 0,
                    '50th': np.percentile(revenues, 50) if revenues else 0,
                    '75th': np.percentile(revenues, 75) if revenues else 0,
                    '90th': np.percentile(revenues, 90) if revenues else 0
                }
            }
            
            # LTV by source analysis
            source_ltv = {}
            for journey in journeys:
                if journey.touchpoints and journey.total_revenue > 0:
                    first_source = journey.touchpoints[0].source
                    if first_source not in source_ltv:
                        source_ltv[first_source] = []
                    source_ltv[first_source].append(journey.total_revenue)
            
            ltv_analysis['ltv_by_source'] = {
                source: {
                    'count': len(revenues),
                    'average_ltv': np.mean(revenues),
                    'total_revenue': sum(revenues)
                }
                for source, revenues in source_ltv.items()
            }
            
            return ltv_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing customer LTV: {e}")
            raise
    
    async def get_multi_touch_attribution_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive multi-touch attribution report"""
        
        try:
            # Calculate attribution using different models
            models = [
                AttributionModel.FIRST_TOUCH,
                AttributionModel.LAST_TOUCH,
                AttributionModel.LINEAR,
                AttributionModel.TIME_DECAY,
                AttributionModel.POSITION_BASED
            ]
            
            attribution_comparison = {}
            
            for model in models:
                results = await self.calculate_attribution(model, days)
                attribution_comparison[model.value] = [
                    {
                        'source': r.source,
                        'medium': r.medium,
                        'campaign': r.campaign,
                        'attributed_revenue': r.attributed_revenue,
                        'attributed_conversions': r.attributed_conversions,
                        'roi': r.roi,
                        'roas': r.roas
                    }
                    for r in results[:10]  # Top 10 for each model
                ]
            
            # Get journey insights
            journey_insights = await self._analyze_customer_journeys(days)
            
            # Get channel performance
            channel_performance = await self._analyze_channel_performance(days)
            
            report = {
                'report_period_days': days,
                'generated_at': datetime.utcnow().isoformat(),
                'attribution_models': attribution_comparison,
                'journey_insights': journey_insights,
                'channel_performance': channel_performance,
                'summary': {
                    'total_attributed_revenue': sum(
                        r['attributed_revenue'] 
                        for r in attribution_comparison.get('linear', [])
                    ),
                    'total_conversions': sum(
                        r['attributed_conversions'] 
                        for r in attribution_comparison.get('linear', [])
                    ),
                    'top_performing_source': max(
                        attribution_comparison.get('linear', []),
                        key=lambda x: x['attributed_revenue'],
                        default={'source': 'N/A'}
                    )['source']
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating attribution report: {e}")
            raise
    
    def _calculate_attribution_weights(self, touchpoints: List[TouchPoint], 
                                     model: AttributionModel) -> Dict[TouchPoint, float]:
        """Calculate attribution weights for touchpoints based on model"""
        
        if not touchpoints:
            return {}
        
        weights = {}
        
        if model == AttributionModel.FIRST_TOUCH:
            weights[touchpoints[0]] = 1.0
            
        elif model == AttributionModel.LAST_TOUCH:
            weights[touchpoints[-1]] = 1.0
            
        elif model == AttributionModel.LINEAR:
            weight_per_touch = 1.0 / len(touchpoints)
            for touchpoint in touchpoints:
                weights[touchpoint] = weight_per_touch
                
        elif model == AttributionModel.TIME_DECAY:
            decay_rate = self.attribution_weights[model]["decay_rate"]
            total_weight = 0
            
            # Calculate weights with time decay (more recent = higher weight)
            for i, touchpoint in enumerate(reversed(touchpoints)):
                weight = decay_rate ** i
                weights[touchpoint] = weight
                total_weight += weight
            
            # Normalize weights to sum to 1
            for touchpoint in weights:
                weights[touchpoint] /= total_weight
                
        elif model == AttributionModel.POSITION_BASED:
            if len(touchpoints) == 1:
                weights[touchpoints[0]] = 1.0
            elif len(touchpoints) == 2:
                weights[touchpoints[0]] = 0.5
                weights[touchpoints[1]] = 0.5
            else:
                # First and last get 40% each, middle touchpoints share 20%
                weights[touchpoints[0]] = 0.4
                weights[touchpoints[-1]] = 0.4
                
                middle_weight = 0.2 / (len(touchpoints) - 2)
                for touchpoint in touchpoints[1:-1]:
                    weights[touchpoint] = middle_weight
        
        return weights
    
    async def _store_touchpoint(self, touchpoint: TouchPoint):
        """Store touchpoint data"""
        
        if self.redis_client:
            touchpoint_data = {
                'touchpoint_id': touchpoint.touchpoint_id,
                'customer_id': touchpoint.customer_id,
                'source': touchpoint.source,
                'medium': touchpoint.medium,
                'campaign': touchpoint.campaign,
                'content': touchpoint.content,
                'timestamp': touchpoint.timestamp.isoformat(),
                'value': touchpoint.value
            }
            
            self.redis_client.hset(
                self.storage_keys['touchpoints'],
                touchpoint.touchpoint_id,
                json.dumps(touchpoint_data)
            )
    
    async def _update_customer_journey(self, touchpoint: TouchPoint):
        """Update customer journey with new touchpoint"""
        
        journey = await self._get_customer_journey(touchpoint.customer_id)
        
        if not journey:
            # Create new journey
            journey = CustomerJourney(
                customer_id=touchpoint.customer_id,
                first_touch=touchpoint.timestamp,
                last_touch=touchpoint.timestamp,
                total_touchpoints=1,
                touchpoints=[touchpoint]
            )
        else:
            # Update existing journey
            journey.touchpoints.append(touchpoint)
            journey.last_touch = touchpoint.timestamp
            journey.total_touchpoints = len(journey.touchpoints)
        
        await self._store_customer_journey(journey)
    
    async def _get_customer_journey(self, customer_id: str) -> Optional[CustomerJourney]:
        """Get customer journey by ID"""
        
        if not self.redis_client:
            return None
        
        journey_data = self.redis_client.hget(self.storage_keys['journeys'], f"journey_{customer_id}")
        
        if journey_data:
            data = json.loads(journey_data)
            
            # Reconstruct touchpoints
            touchpoints = []
            for tp_data in data.get('touchpoints', []):
                touchpoint = TouchPoint(
                    touchpoint_id=tp_data['touchpoint_id'],
                    customer_id=tp_data['customer_id'],
                    source=tp_data['source'],
                    medium=tp_data['medium'],
                    campaign=tp_data['campaign'],
                    content=tp_data['content'],
                    timestamp=datetime.fromisoformat(tp_data['timestamp']),
                    value=tp_data['value']
                )
                touchpoints.append(touchpoint)
            
            journey = CustomerJourney(
                customer_id=data['customer_id'],
                first_touch=datetime.fromisoformat(data['first_touch']),
                last_touch=datetime.fromisoformat(data['last_touch']),
                total_touchpoints=data['total_touchpoints'],
                touchpoints=touchpoints,
                total_revenue=data.get('total_revenue', 0.0),
                conversion_date=datetime.fromisoformat(data['conversion_date']) if data.get('conversion_date') else None,
                journey_duration_days=data.get('journey_duration_days', 0)
            )
            
            return journey
        
        return None
    
    def _generate_touchpoint_id(self) -> str:
        """Generate unique touchpoint ID"""
        import uuid
        return f"tp_{uuid.uuid4().hex[:8]}"

# Global attribution engine instance
attribution_engine = RevenueAttributionEngine()