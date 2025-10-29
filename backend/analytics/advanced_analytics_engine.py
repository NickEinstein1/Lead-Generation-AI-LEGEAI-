"""
Advanced Analytics Engine for Insurance Lead Scoring Platform

Comprehensive analytics system providing real-time insights, predictive analytics,
sales performance tracking, and revenue attribution analysis.
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import redis
import json

logger = logging.getLogger(__name__)

class MetricType(Enum):
    CONVERSION_RATE = "conversion_rate"
    ROI = "roi"
    LTV = "lifetime_value"
    LEAD_VELOCITY = "lead_velocity"
    COST_PER_ACQUISITION = "cpa"
    REVENUE_PER_LEAD = "rpl"

class TimeGranularity(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

@dataclass
class AnalyticsMetrics:
    """Core analytics metrics"""
    total_leads: int = 0
    converted_leads: int = 0
    conversion_rate: float = 0.0
    total_revenue: float = 0.0
    total_cost: float = 0.0
    roi: float = 0.0
    average_ltv: float = 0.0
    cost_per_acquisition: float = 0.0
    revenue_per_lead: float = 0.0
    lead_velocity: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class LeadSourceMetrics:
    """Lead source performance metrics"""
    source_id: str
    source_name: str
    total_leads: int = 0
    converted_leads: int = 0
    conversion_rate: float = 0.0
    total_revenue: float = 0.0
    cost_per_lead: float = 0.0
    roi: float = 0.0
    average_lead_score: float = 0.0
    average_ltv: float = 0.0

@dataclass
class SalesRepMetrics:
    """Sales representative performance metrics"""
    rep_id: str
    rep_name: str
    leads_assigned: int = 0
    leads_contacted: int = 0
    leads_converted: int = 0
    conversion_rate: float = 0.0
    average_response_time: float = 0.0
    total_revenue: float = 0.0
    calls_made: int = 0
    emails_sent: int = 0
    meetings_scheduled: int = 0
    pipeline_value: float = 0.0

class AdvancedAnalyticsEngine:
    """Advanced analytics engine with real-time insights and predictions"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Predictive models
        self.conversion_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ltv_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.seasonal_model = LinearRegression()
        
        # Model training status
        self.models_trained = False
        self.last_model_update = None
        
        # Cache for real-time metrics
        self.metrics_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize data storage
        self._initialize_data_structures()
    
    def _initialize_data_structures(self):
        """Initialize data structures for analytics"""
        
        # Time series data keys
        self.ts_keys = {
            'leads_hourly': 'analytics:leads:hourly',
            'conversions_hourly': 'analytics:conversions:hourly',
            'revenue_hourly': 'analytics:revenue:hourly',
            'lead_sources': 'analytics:lead_sources',
            'sales_reps': 'analytics:sales_reps',
            'campaigns': 'analytics:campaigns'
        }
        
        logger.info("Analytics data structures initialized")
    
    async def track_lead_event(self, event_data: Dict[str, Any]):
        """Track lead-related events for analytics"""
        
        try:
            event_type = event_data.get('event_type')
            timestamp = datetime.utcnow()
            hour_key = timestamp.strftime('%Y-%m-%d:%H')
            
            # Track different event types
            if event_type == 'lead_created':
                await self._track_lead_creation(event_data, hour_key)
            elif event_type == 'lead_converted':
                await self._track_conversion(event_data, hour_key)
            elif event_type == 'lead_contacted':
                await self._track_contact_event(event_data)
            elif event_type == 'revenue_generated':
                await self._track_revenue(event_data, hour_key)
            
            logger.debug(f"Tracked {event_type} event")
            
        except Exception as e:
            logger.error(f"Error tracking lead event: {e}")
    
    async def _track_lead_creation(self, event_data: Dict[str, Any], hour_key: str):
        """Track lead creation events"""
        
        # Increment hourly lead count
        await self._increment_time_series('leads_hourly', hour_key)
        
        # Track by source
        source_id = event_data.get('source_id', 'unknown')
        await self._update_source_metrics(source_id, 'leads_created', 1)
        
        # Track lead score distribution
        lead_score = event_data.get('lead_score', 0)
        await self._track_score_distribution(lead_score, hour_key)
    
    async def _track_conversion(self, event_data: Dict[str, Any], hour_key: str):
        """Track conversion events"""
        
        # Increment hourly conversion count
        await self._increment_time_series('conversions_hourly', hour_key)
        
        # Track by source
        source_id = event_data.get('source_id', 'unknown')
        await self._update_source_metrics(source_id, 'conversions', 1)
        
        # Track by sales rep
        rep_id = event_data.get('rep_id')
        if rep_id:
            await self._update_rep_metrics(rep_id, 'conversions', 1)
    
    async def _track_revenue(self, event_data: Dict[str, Any], hour_key: str):
        """Track revenue events"""
        
        revenue = event_data.get('revenue', 0)
        
        # Track hourly revenue
        current_revenue = float(self.redis_client.hget(self.ts_keys['revenue_hourly'], hour_key) or 0)
        self.redis_client.hset(self.ts_keys['revenue_hourly'], hour_key, current_revenue + revenue)
        
        # Track by source
        source_id = event_data.get('source_id', 'unknown')
        await self._update_source_metrics(source_id, 'revenue', revenue)
        
        # Track by sales rep
        rep_id = event_data.get('rep_id')
        if rep_id:
            await self._update_rep_metrics(rep_id, 'revenue', revenue)
    
    async def _increment_time_series(self, series_name: str, time_key: str, value: int = 1):
        """Increment time series counter"""
        
        key = self.ts_keys[series_name]
        current_value = int(self.redis_client.hget(key, time_key) or 0)
        self.redis_client.hset(key, time_key, current_value + value)
        
        # Set expiration for old data (keep 90 days)
        self.redis_client.expire(key, 90 * 24 * 3600)
    
    async def get_real_time_metrics(self, time_range: str = "24h") -> AnalyticsMetrics:
        """Get real-time analytics metrics"""
        
        cache_key = f"metrics:{time_range}"
        
        # Check cache first
        if cache_key in self.metrics_cache:
            cached_time, cached_data = self.metrics_cache[cache_key]
            if (datetime.utcnow() - cached_time).seconds < self.cache_ttl:
                return cached_data
        
        # Calculate metrics
        end_time = datetime.utcnow()
        if time_range == "24h":
            start_time = end_time - timedelta(hours=24)
        elif time_range == "7d":
            start_time = end_time - timedelta(days=7)
        elif time_range == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(hours=24)
        
        # Get time series data
        leads_data = await self._get_time_series_data('leads_hourly', start_time, end_time)
        conversions_data = await self._get_time_series_data('conversions_hourly', start_time, end_time)
        revenue_data = await self._get_time_series_data('revenue_hourly', start_time, end_time)
        
        # Calculate metrics
        total_leads = sum(leads_data.values())
        converted_leads = sum(conversions_data.values())
        total_revenue = sum(revenue_data.values())
        
        conversion_rate = (converted_leads / total_leads * 100) if total_leads > 0 else 0
        revenue_per_lead = total_revenue / total_leads if total_leads > 0 else 0
        
        # Calculate lead velocity (leads per hour)
        hours = max(1, (end_time - start_time).total_seconds() / 3600)
        lead_velocity = total_leads / hours
        
        metrics = AnalyticsMetrics(
            total_leads=total_leads,
            converted_leads=converted_leads,
            conversion_rate=conversion_rate,
            total_revenue=total_revenue,
            revenue_per_lead=revenue_per_lead,
            lead_velocity=lead_velocity,
            timestamp=datetime.utcnow()
        )
        
        # Cache the result
        self.metrics_cache[cache_key] = (datetime.utcnow(), metrics)
        
        return metrics
    
    async def get_lead_source_performance(self, time_range: str = "30d") -> List[LeadSourceMetrics]:
        """Get lead source performance analysis"""
        
        sources_data = self.redis_client.hgetall(self.ts_keys['lead_sources'])
        source_metrics = []
        
        for source_id, data_str in sources_data.items():
            try:
                data = json.loads(data_str)
                
                metrics = LeadSourceMetrics(
                    source_id=source_id,
                    source_name=data.get('name', source_id),
                    total_leads=data.get('leads_created', 0),
                    converted_leads=data.get('conversions', 0),
                    total_revenue=data.get('revenue', 0),
                    cost_per_lead=data.get('cost_per_lead', 0),
                    average_lead_score=data.get('avg_lead_score', 0)
                )
                
                # Calculate derived metrics
                if metrics.total_leads > 0:
                    metrics.conversion_rate = (metrics.converted_leads / metrics.total_leads) * 100
                    metrics.average_ltv = metrics.total_revenue / metrics.converted_leads if metrics.converted_leads > 0 else 0
                    
                    total_cost = metrics.cost_per_lead * metrics.total_leads
                    metrics.roi = ((metrics.total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
                
                source_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error processing source {source_id}: {e}")
        
        # Sort by ROI descending
        source_metrics.sort(key=lambda x: x.roi, reverse=True)
        
        return source_metrics
    
    async def get_sales_rep_performance(self, time_range: str = "30d") -> List[SalesRepMetrics]:
        """Get sales representative performance analysis"""
        
        reps_data = self.redis_client.hgetall(self.ts_keys['sales_reps'])
        rep_metrics = []
        
        for rep_id, data_str in reps_data.items():
            try:
                data = json.loads(data_str)
                
                metrics = SalesRepMetrics(
                    rep_id=rep_id,
                    rep_name=data.get('name', rep_id),
                    leads_assigned=data.get('leads_assigned', 0),
                    leads_contacted=data.get('leads_contacted', 0),
                    leads_converted=data.get('conversions', 0),
                    total_revenue=data.get('revenue', 0),
                    calls_made=data.get('calls_made', 0),
                    emails_sent=data.get('emails_sent', 0),
                    meetings_scheduled=data.get('meetings_scheduled', 0),
                    average_response_time=data.get('avg_response_time', 0),
                    pipeline_value=data.get('pipeline_value', 0)
                )
                
                # Calculate conversion rate
                if metrics.leads_assigned > 0:
                    metrics.conversion_rate = (metrics.leads_converted / metrics.leads_assigned) * 100
                
                rep_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error processing rep {rep_id}: {e}")
        
        # Sort by conversion rate descending
        rep_metrics.sort(key=lambda x: x.conversion_rate, reverse=True)
        
        return rep_metrics
    
    async def generate_predictive_insights(self, forecast_days: int = 30) -> Dict[str, Any]:
        """Generate predictive analytics insights"""
        
        if not self.models_trained:
            await self._train_predictive_models()
        
        # Get historical data for prediction
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=90)  # Use 90 days of history
        
        historical_data = await self._prepare_prediction_data(start_time, end_time)
        
        if len(historical_data) < 30:  # Need minimum data
            return {"error": "Insufficient historical data for predictions"}
        
        # Generate forecasts
        lead_forecast = await self._forecast_leads(historical_data, forecast_days)
        conversion_forecast = await self._forecast_conversions(historical_data, forecast_days)
        revenue_forecast = await self._forecast_revenue(historical_data, forecast_days)
        seasonal_insights = await self._analyze_seasonal_patterns(historical_data)
        
        return {
            "forecast_period_days": forecast_days,
            "lead_volume_forecast": lead_forecast,
            "conversion_forecast": conversion_forecast,
            "revenue_forecast": revenue_forecast,
            "seasonal_insights": seasonal_insights,
            "confidence_intervals": {
                "leads": {"lower": lead_forecast * 0.85, "upper": lead_forecast * 1.15},
                "conversions": {"lower": conversion_forecast * 0.8, "upper": conversion_forecast * 1.2},
                "revenue": {"lower": revenue_forecast * 0.75, "upper": revenue_forecast * 1.25}
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _train_predictive_models(self):
        """Train predictive models with historical data"""
        
        try:
            # Get training data (last 6 months)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=180)
            
            training_data = await self._prepare_prediction_data(start_time, end_time)
            
            if len(training_data) < 50:  # Need minimum training data
                logger.warning("Insufficient data for model training")
                return
            
            # Prepare features and targets
            df = pd.DataFrame(training_data)
            
            # Feature engineering
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Rolling averages
            df['leads_7d_avg'] = df['leads'].rolling(7, min_periods=1).mean()
            df['conversions_7d_avg'] = df['conversions'].rolling(7, min_periods=1).mean()
            
            features = ['day_of_week', 'month', 'is_weekend', 'leads_7d_avg', 'conversions_7d_avg']
            X = df[features].fillna(0)
            
            # Train models
            if 'leads' in df.columns:
                self.conversion_model.fit(X, df['leads'])
            
            if 'revenue' in df.columns:
                self.ltv_model.fit(X, df['revenue'])
            
            self.models_trained = True
            self.last_model_update = datetime.utcnow()
            
            logger.info("Predictive models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training predictive models: {e}")
    
    async def _get_time_series_data(self, series_name: str, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Get time series data for a date range"""
        
        key = self.ts_keys[series_name]
        all_data = self.redis_client.hgetall(key)
        
        filtered_data = {}
        for time_key, value in all_data.items():
            try:
                # Parse time key (format: YYYY-MM-DD:HH)
                dt = datetime.strptime(time_key, '%Y-%m-%d:%H')
                if start_time <= dt <= end_time:
                    filtered_data[time_key] = float(value)
            except ValueError:
                continue
        
        return filtered_data
    
    async def _update_source_metrics(self, source_id: str, metric: str, value: float):
        """Update lead source metrics"""
        
        key = self.ts_keys['lead_sources']
        current_data = self.redis_client.hget(key, source_id)
        
        if current_data:
            data = json.loads(current_data)
        else:
            data = {'name': source_id}
        
        data[metric] = data.get(metric, 0) + value
        
        self.redis_client.hset(key, source_id, json.dumps(data))
    
    async def _update_rep_metrics(self, rep_id: str, metric: str, value: float):
        """Update sales rep metrics"""
        
        key = self.ts_keys['sales_reps']
        current_data = self.redis_client.hget(key, rep_id)
        
        if current_data:
            data = json.loads(current_data)
        else:
            data = {'name': rep_id}
        
        data[metric] = data.get(metric, 0) + value
        
        self.redis_client.hset(key, rep_id, json.dumps(data))

# Global analytics engine instance
analytics_engine = AdvancedAnalyticsEngine()