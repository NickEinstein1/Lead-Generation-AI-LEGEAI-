"""
Advanced Analytics Engine for Insurance Lead Scoring Platform

Provides comprehensive analytics, reporting, and business intelligence capabilities
including predictive analytics, trend analysis, and automated insights generation.
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
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"

class MetricType(Enum): # TODO: Expand this list 
    CONVERSION_RATE = "conversion_rate"
    REVENUE_PER_LEAD = "revenue_per_lead"
    LEAD_QUALITY_SCORE = "lead_quality_score"
    COST_PER_ACQUISITION = "cost_per_acquisition"
    LIFETIME_VALUE = "lifetime_value"
    CHURN_RATE = "churn_rate"
    ENGAGEMENT_SCORE = "engagement_score"

class TimeGranularity(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

@dataclass
class AnalyticsQuery:
    query_id: str
    query_type: AnalyticsType
    metrics: List[MetricType]
    time_range: Tuple[datetime, datetime]
    granularity: TimeGranularity
    filters: Dict[str, Any] = field(default_factory=dict)
    segments: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AnalyticsResult:
    query_id: str
    data: Dict[str, Any]
    insights: List[str]
    visualizations: Dict[str, Any]
    confidence_score: float
    generated_at: datetime = field(default_factory=datetime.utcnow)

class AdvancedAnalyticsEngine:
    """Advanced analytics engine with ML-powered insights"""
    
    def __init__(self):
        self.queries = {}
        self.results_cache = {}
        self.ml_models = {}
        self.anomaly_detectors = {}
        self.trend_analyzers = {}
        
    async def execute_query(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Execute analytics query and generate insights"""
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cache_key in self.results_cache:
                cached_result = self.results_cache[cache_key]
                if self._is_cache_valid(cached_result, query):
                    return cached_result
            
            # Execute query based on type
            if query.query_type == AnalyticsType.DESCRIPTIVE:
                result = await self._execute_descriptive_analytics(query)
            elif query.query_type == AnalyticsType.DIAGNOSTIC:
                result = await self._execute_diagnostic_analytics(query)
            elif query.query_type == AnalyticsType.PREDICTIVE:
                result = await self._execute_predictive_analytics(query)
            elif query.query_type == AnalyticsType.PRESCRIPTIVE:
                result = await self._execute_prescriptive_analytics(query)
            else:
                raise ValueError(f"Unsupported query type: {query.query_type}")
            
            # Cache result
            self.results_cache[cache_key] = result
            self.queries[query.query_id] = query
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing analytics query {query.query_id}: {e}")
            raise
    
    async def _execute_descriptive_analytics(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Execute descriptive analytics"""
        
        # Get data for the specified time range
        data = await self._fetch_data(query)
        
        # Calculate basic statistics
        stats = self._calculate_descriptive_stats(data, query.metrics)
        
        # Generate time series analysis
        time_series = self._generate_time_series(data, query.granularity)
        
        # Segment analysis
        segment_analysis = self._analyze_segments(data, query.segments) if query.segments else {}
        
        # Generate insights
        insights = self._generate_descriptive_insights(stats, time_series, segment_analysis)
        
        # Create visualizations
        visualizations = self._create_descriptive_visualizations(stats, time_series, segment_analysis)
        
        return AnalyticsResult(
            query_id=query.query_id,
            data={
                'statistics': stats,
                'time_series': time_series,
                'segments': segment_analysis
            },
            insights=insights,
            visualizations=visualizations,
            confidence_score=0.95
        )
    
    async def _execute_diagnostic_analytics(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Execute diagnostic analytics to understand why things happened"""
        
        data = await self._fetch_data(query)
        
        # Correlation analysis
        correlations = self._analyze_correlations(data, query.metrics)
        
        # Anomaly detection
        anomalies = self._detect_anomalies(data, query.metrics)
        
        # Root cause analysis
        root_causes = self._analyze_root_causes(data, anomalies)
        
        # Performance drivers analysis
        drivers = self._analyze_performance_drivers(data, query.metrics)
        
        insights = self._generate_diagnostic_insights(correlations, anomalies, root_causes, drivers)
        visualizations = self._create_diagnostic_visualizations(correlations, anomalies, drivers)
        
        return AnalyticsResult(
            query_id=query.query_id,
            data={
                'correlations': correlations,
                'anomalies': anomalies,
                'root_causes': root_causes,
                'performance_drivers': drivers
            },
            insights=insights,
            visualizations=visualizations,
            confidence_score=0.88
        )
    
    async def _execute_predictive_analytics(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Execute predictive analytics"""
        
        data = await self._fetch_data(query)
        
        # Time series forecasting
        forecasts = self._generate_forecasts(data, query.metrics, query.granularity)
        
        # Lead scoring predictions
        lead_predictions = self._predict_lead_outcomes(data)
        
        # Churn prediction
        churn_predictions = self._predict_churn(data)
        
        # Revenue forecasting
        revenue_forecast = self._forecast_revenue(data, query.time_range)
        
        insights = self._generate_predictive_insights(forecasts, lead_predictions, churn_predictions)
        visualizations = self._create_predictive_visualizations(forecasts, lead_predictions, revenue_forecast)
        
        return AnalyticsResult(
            query_id=query.query_id,
            data={
                'forecasts': forecasts,
                'lead_predictions': lead_predictions,
                'churn_predictions': churn_predictions,
                'revenue_forecast': revenue_forecast
            },
            insights=insights,
            visualizations=visualizations,
            confidence_score=0.82
        )
    
    async def _execute_prescriptive_analytics(self, query: AnalyticsQuery) -> AnalyticsResult:
        """Execute prescriptive analytics to recommend actions"""
        
        data = await self._fetch_data(query)
        
        # Optimization recommendations
        optimizations = self._generate_optimization_recommendations(data, query.metrics)
        
        # Resource allocation suggestions
        resource_allocation = self._optimize_resource_allocation(data)
        
        # Campaign optimization
        campaign_recommendations = self._optimize_campaigns(data)
        
        # Risk mitigation strategies
        risk_strategies = self._generate_risk_mitigation_strategies(data)
        
        insights = self._generate_prescriptive_insights(optimizations, resource_allocation, campaign_recommendations)
        visualizations = self._create_prescriptive_visualizations(optimizations, resource_allocation)
        
        return AnalyticsResult(
            query_id=query.query_id,
            data={
                'optimizations': optimizations,
                'resource_allocation': resource_allocation,
                'campaign_recommendations': campaign_recommendations,
                'risk_strategies': risk_strategies
            },
            insights=insights,
            visualizations=visualizations,
            confidence_score=0.75
        )
    
    def _calculate_descriptive_stats(self, data: pd.DataFrame, metrics: List[MetricType]) -> Dict[str, Any]:
        """Calculate descriptive statistics"""
        
        stats = {}
        
        for metric in metrics:
            if metric.value in data.columns:
                column_data = data[metric.value].dropna()
                
                stats[metric.value] = {
                    'count': len(column_data),
                    'mean': float(column_data.mean()),
                    'median': float(column_data.median()),
                    'std': float(column_data.std()),
                    'min': float(column_data.min()),
                    'max': float(column_data.max()),
                    'q25': float(column_data.quantile(0.25)),
                    'q75': float(column_data.quantile(0.75)),
                    'skewness': float(column_data.skew()),
                    'kurtosis': float(column_data.kurtosis())
                }
        
        return stats
    
    def _generate_time_series(self, data: pd.DataFrame, granularity: TimeGranularity) -> Dict[str, Any]:
        """Generate time series analysis"""
        
        if 'timestamp' not in data.columns:
            return {}
        
        # Group by time granularity
        if granularity == TimeGranularity.HOURLY:
            data['period'] = data['timestamp'].dt.floor('H')
        elif granularity == TimeGranularity.DAILY:
            data['period'] = data['timestamp'].dt.date
        elif granularity == TimeGranularity.WEEKLY:
            data['period'] = data['timestamp'].dt.to_period('W')
        elif granularity == TimeGranularity.MONTHLY:
            data['period'] = data['timestamp'].dt.to_period('M')
        else:
            data['period'] = data['timestamp'].dt.date
        
        # Aggregate metrics by period
        time_series = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column != 'timestamp':
                series_data = data.groupby('period')[column].agg(['count', 'mean', 'sum']).reset_index()
                time_series[column] = {
                    'periods': [str(p) for p in series_data['period']],
                    'counts': series_data['count'].tolist(),
                    'means': series_data['mean'].tolist(),
                    'sums': series_data['sum'].tolist()
                }
        
        return time_series
    
    def _analyze_correlations(self, data: pd.DataFrame, metrics: List[MetricType]) -> Dict[str, Any]:
        """Analyze correlations between metrics"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'metric1': correlation_matrix.columns[i],
                        'metric2': correlation_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def _detect_anomalies(self, data: pd.DataFrame, metrics: List[MetricType]) -> Dict[str, Any]:
        """Detect anomalies in the data"""
        
        anomalies = {}
        
        for metric in metrics:
            if metric.value in data.columns:
                column_data = data[metric.value].dropna()
                
                if len(column_data) > 10:  # Need sufficient data
                    # Use Isolation Forest for anomaly detection
                    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_labels = isolation_forest.fit_predict(column_data.values.reshape(-1, 1))
                    
                    anomaly_indices = np.where(anomaly_labels == -1)[0]
                    anomaly_values = column_data.iloc[anomaly_indices].tolist()
                    
                    anomalies[metric.value] = {
                        'count': len(anomaly_indices),
                        'indices': anomaly_indices.tolist(),
                        'values': anomaly_values,
                        'percentage': len(anomaly_indices) / len(column_data) * 100
                    }
        
        return anomalies

# Global analytics engine instance
analytics_engine = AdvancedAnalyticsEngine()