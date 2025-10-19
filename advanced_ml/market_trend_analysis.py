"""
Market Trend Analysis Engine

Advanced market trend analysis for adapting lead scoring based on
market conditions, seasonal patterns, and economic indicators.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import yfinance as yf
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    RECESSION = "recession"
    RECOVERY = "recovery"

class SeasonalPattern(Enum):
    Q1_STRONG = "q1_strong"
    Q2_MODERATE = "q2_moderate"
    Q3_SLOW = "q3_slow"
    Q4_PEAK = "q4_peak"
    HOLIDAY_IMPACT = "holiday_impact"
    YEAR_END_RUSH = "year_end_rush"

class TrendDirection(Enum):
    UPWARD = "upward"
    DOWNWARD = "downward"
    STABLE = "stable"
    CYCLICAL = "cyclical"

@dataclass
class MarketIndicators:
    """Market indicators and metrics"""
    # Economic indicators
    gdp_growth: float = 0.0
    unemployment_rate: float = 0.0
    inflation_rate: float = 0.0
    interest_rates: float = 0.0
    consumer_confidence: float = 0.0
    
    # Market indicators
    stock_market_performance: float = 0.0
    insurance_sector_performance: float = 0.0
    volatility_index: float = 0.0
    
    # Industry specific
    insurance_penetration: float = 0.0
    regulatory_changes: float = 0.0
    digital_adoption_rate: float = 0.0
    
    # Seasonal factors
    month: int = 1
    quarter: int = 1
    is_holiday_season: bool = False
    is_tax_season: bool = False
    
    # Competitive landscape
    competitor_activity: float = 0.0
    market_saturation: float = 0.0
    new_entrants: int = 0

@dataclass
class TrendAnalysis:
    """Market trend analysis result"""
    market_condition: MarketCondition
    seasonal_pattern: SeasonalPattern
    trend_direction: TrendDirection
    trend_strength: float
    volatility_score: float
    opportunity_score: float
    risk_score: float
    recommended_adjustments: Dict[str, float]
    confidence_level: float
    analysis_timestamp: datetime = field(default_factory=datetime.now)

class MarketTrendAnalysisEngine:
    """Advanced market trend analysis and scoring adaptation"""
    
    def __init__(self):
        self.market_data = {}
        self.trend_models = {}
        self.seasonal_patterns = {}
        self.economic_indicators = {}
        
        # Scoring adjustments
        self.base_adjustments = {
            'lead_score_multiplier': 1.0,
            'urgency_factor': 1.0,
            'follow_up_frequency': 1.0,
            'conversion_probability': 1.0
        }
        
        # Data sources
        self.data_sources = {
            'economic': 'https://api.stlouisfed.org/fred/series/observations',
            'market': 'yahoo_finance',
            'insurance': 'custom_api',
            'seasonal': 'internal_data'
        }
        
        logger.info("Market Trend Analysis Engine initialized")
    
    async def collect_market_data(self) -> MarketIndicators:
        """Collect comprehensive market data from various sources"""
        
        try:
            logger.info("Collecting market data...")
            
            # Collect economic indicators
            economic_data = await self._collect_economic_indicators()
            
            # Collect market data
            market_data = await self._collect_market_indicators()
            
            # Collect industry data
            industry_data = await self._collect_industry_indicators()
            
            # Collect seasonal data
            seasonal_data = self._collect_seasonal_indicators()
            
            # Combine all indicators
            indicators = MarketIndicators(
                **economic_data,
                **market_data,
                **industry_data,
                **seasonal_data
            )
            
            # Store for analysis
            self.market_data[datetime.now().date()] = indicators
            
            logger.info("Market data collection completed")
            return indicators
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            # Return default indicators
            return MarketIndicators()
    
    async def _collect_economic_indicators(self) -> Dict[str, float]:
        """Collect economic indicators from FRED API"""
        
        try:
            # Simulated economic data (replace with actual API calls)
            economic_data = {
                'gdp_growth': 2.1,  # Annual GDP growth rate
                'unemployment_rate': 3.7,  # Unemployment rate
                'inflation_rate': 3.2,  # CPI inflation rate
                'interest_rates': 5.25,  # Federal funds rate
                'consumer_confidence': 102.3  # Consumer confidence index
            }
            
            return economic_data
            
        except Exception as e:
            logger.error(f"Error collecting economic indicators: {e}")
            return {
                'gdp_growth': 2.0,
                'unemployment_rate': 4.0,
                'inflation_rate': 3.0,
                'interest_rates': 5.0,
                'consumer_confidence': 100.0
            }
    
    async def _collect_market_indicators(self) -> Dict[str, float]:
        """Collect market indicators from financial APIs"""
        
        try:
            # Get S&P 500 performance
            sp500 = yf.Ticker("^GSPC")
            sp500_data = sp500.history(period="1mo")
            sp500_performance = ((sp500_data['Close'][-1] - sp500_data['Close'][0]) / sp500_data['Close'][0]) * 100
            
            # Get insurance sector ETF performance
            insurance_etf = yf.Ticker("KIE")  # SPDR S&P Insurance ETF
            insurance_data = insurance_etf.history(period="1mo")
            insurance_performance = ((insurance_data['Close'][-1] - insurance_data['Close'][0]) / insurance_data['Close'][0]) * 100
            
            # Get VIX (volatility index)
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            volatility_index = vix_data['Close'][-1]
            
            return {
                'stock_market_performance': float(sp500_performance),
                'insurance_sector_performance': float(insurance_performance),
                'volatility_index': float(volatility_index)
            }
            
        except Exception as e:
            logger.error(f"Error collecting market indicators: {e}")
            return {
                'stock_market_performance': 1.5,
                'insurance_sector_performance': 2.0,
                'volatility_index': 20.0
            }
    
    async def _collect_industry_indicators(self) -> Dict[str, float]:
        """Collect insurance industry specific indicators"""
        
        try:
            # Simulated industry data (replace with actual industry APIs)
            industry_data = {
                'insurance_penetration': 7.2,  # Insurance penetration rate
                'regulatory_changes': 0.3,  # Regulatory change impact score
                'digital_adoption_rate': 68.5,  # Digital adoption percentage
                'competitor_activity': 0.7,  # Competitor activity score
                'market_saturation': 0.6,  # Market saturation level
                'new_entrants': 3  # Number of new market entrants
            }
            
            return industry_data
            
        except Exception as e:
            logger.error(f"Error collecting industry indicators: {e}")
            return {
                'insurance_penetration': 7.0,
                'regulatory_changes': 0.5,
                'digital_adoption_rate': 65.0,
                'competitor_activity': 0.5,
                'market_saturation': 0.5,
                'new_entrants': 2
            }
    
    def _collect_seasonal_indicators(self) -> Dict[str, Any]:
        """Collect seasonal and calendar-based indicators"""
        
        now = datetime.now()
        
        # Determine seasonal factors
        is_holiday_season = now.month in [11, 12, 1]
        is_tax_season = now.month in [1, 2, 3, 4]
        
        return {
            'month': now.month,
            'quarter': (now.month - 1) // 3 + 1,
            'is_holiday_season': is_holiday_season,
            'is_tax_season': is_tax_season
        }
    
    async def analyze_market_trends(self, indicators: MarketIndicators) -> TrendAnalysis:
        """Analyze market trends and determine conditions"""
        
        try:
            logger.info("Analyzing market trends...")
            
            # Determine market condition
            market_condition = self._determine_market_condition(indicators)
            
            # Identify seasonal pattern
            seasonal_pattern = self._identify_seasonal_pattern(indicators)
            
            # Analyze trend direction
            trend_direction, trend_strength = self._analyze_trend_direction(indicators)
            
            # Calculate volatility score
            volatility_score = self._calculate_volatility_score(indicators)
            
            # Calculate opportunity and risk scores
            opportunity_score = self._calculate_opportunity_score(indicators, market_condition)
            risk_score = self._calculate_risk_score(indicators, market_condition)
            
            # Generate recommended adjustments
            adjustments = self._generate_scoring_adjustments(
                market_condition, seasonal_pattern, trend_direction, 
                opportunity_score, risk_score
            )
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(indicators)
            
            analysis = TrendAnalysis(
                market_condition=market_condition,
                seasonal_pattern=seasonal_pattern,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                volatility_score=volatility_score,
                opportunity_score=opportunity_score,
                risk_score=risk_score,
                recommended_adjustments=adjustments,
                confidence_level=confidence_level
            )
            
            logger.info(f"Market trend analysis completed. Condition: {market_condition.value}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market trends: {e}")
            # Return default analysis
            return TrendAnalysis(
                market_condition=MarketCondition.SIDEWAYS,
                seasonal_pattern=SeasonalPattern.Q2_MODERATE,
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.5,
                volatility_score=0.5,
                opportunity_score=0.5,
                risk_score=0.5,
                recommended_adjustments=self.base_adjustments.copy(),
                confidence_level=0.6
            )
    
    def _determine_market_condition(self, indicators: MarketIndicators) -> MarketCondition:
        """Determine overall market condition"""
        
        # Economic health score
        economic_score = 0
        if indicators.gdp_growth > 2.0:
            economic_score += 1
        if indicators.unemployment_rate < 4.0:
            economic_score += 1
        if indicators.inflation_rate < 4.0:
            economic_score += 1
        if indicators.consumer_confidence > 100:
            economic_score += 1
        
        # Market performance score
        market_score = 0
        if indicators.stock_market_performance > 5:
            market_score += 2
        elif indicators.stock_market_performance > 0:
            market_score += 1
        elif indicators.stock_market_performance < -10:
            market_score -= 2
        elif indicators.stock_market_performance < -5:
            market_score -= 1
        
        # Volatility assessment
        high_volatility = indicators.volatility_index > 30
        
        # Determine condition
        total_score = economic_score + market_score
        
        if total_score >= 5 and not high_volatility:
            return MarketCondition.BULL_MARKET
        elif total_score <= 1 or indicators.gdp_growth < 0:
            return MarketCondition.RECESSION if indicators.gdp_growth < -1 else MarketCondition.BEAR_MARKET
        elif high_volatility:
            return MarketCondition.VOLATILE
        elif total_score >= 3:
            return MarketCondition.RECOVERY
        else:
            return MarketCondition.SIDEWAYS
    
    def _identify_seasonal_pattern(self, indicators: MarketIndicators) -> SeasonalPattern:
        """Identify seasonal patterns"""
        
        if indicators.is_holiday_season:
            return SeasonalPattern.HOLIDAY_IMPACT
        elif indicators.quarter == 4 and not indicators.is_holiday_season:
            return SeasonalPattern.YEAR_END_RUSH
        elif indicators.quarter == 1:
            return SeasonalPattern.Q1_STRONG
        elif indicators.quarter == 2:
            return SeasonalPattern.Q2_MODERATE
        else:
            return SeasonalPattern.Q3_SLOW
    
    def _analyze_trend_direction(self, indicators: MarketIndicators) -> Tuple[TrendDirection, float]:
        """Analyze trend direction and strength"""
        
        # Simple trend analysis based on multiple indicators
        positive_indicators = 0
        negative_indicators = 0
        
        # Economic trends
        if indicators.gdp_growth > 2.5:
            positive_indicators += 1
        elif indicators.gdp_growth < 1.0:
            negative_indicators += 1
        
        if indicators.unemployment_rate < 3.5:
            positive_indicators += 1
        elif indicators.unemployment_rate > 5.0:
            negative_indicators += 1
        
        # Market trends
        if indicators.stock_market_performance > 3:
            positive_indicators += 1
        elif indicators.stock_market_performance < -3:
            negative_indicators += 1
        
        if indicators.insurance_sector_performance > 2:
            positive_indicators += 1
        elif indicators.insurance_sector_performance < -2:
            negative_indicators += 1
        
        # Determine direction and strength
        net_trend = positive_indicators - negative_indicators
        strength = abs(net_trend) / 4.0  # Normalize to 0-1
        
        if net_trend > 1:
            return TrendDirection.UPWARD, strength
        elif net_trend < -1:
            return TrendDirection.DOWNWARD, strength
        elif indicators.volatility_index > 25:
            return TrendDirection.CYCLICAL, strength
        else:
            return TrendDirection.STABLE, strength
    
    def _calculate_volatility_score(self, indicators: MarketIndicators) -> float:
        """Calculate market volatility score"""
        
        # Normalize VIX to 0-1 scale (typical range 10-80)
        vix_normalized = min(indicators.volatility_index / 80.0, 1.0)
        
        # Factor in other volatility indicators
        economic_volatility = 0
        if indicators.inflation_rate > 5.0:
            economic_volatility += 0.2
        if indicators.interest_rates > 6.0:
            economic_volatility += 0.2
        
        market_volatility = abs(indicators.stock_market_performance) / 20.0  # Normalize large moves
        
        total_volatility = (vix_normalized * 0.6 + economic_volatility * 0.2 + market_volatility * 0.2)
        return min(total_volatility, 1.0)
    
    def _calculate_opportunity_score(self, indicators: MarketIndicators, 
                                   market_condition: MarketCondition) -> float:
        """Calculate market opportunity score"""
        
        opportunity_score = 0.5  # Base score
        
        # Market condition adjustments
        condition_adjustments = {
            MarketCondition.BULL_MARKET: 0.3,
            MarketCondition.RECOVERY: 0.2,
            MarketCondition.SIDEWAYS: 0.0,
            MarketCondition.VOLATILE: -0.1,
            MarketCondition.BEAR_MARKET: -0.2,
            MarketCondition.RECESSION: -0.3
        }
        opportunity_score += condition_adjustments.get(market_condition, 0)
        
        # Economic indicators
        if indicators.consumer_confidence > 110:
            opportunity_score += 0.1
        elif indicators.consumer_confidence < 90:
            opportunity_score -= 0.1
        
        # Industry specific
        if indicators.digital_adoption_rate > 70:
            opportunity_score += 0.1
        if indicators.insurance_penetration < 8:  # Room for growth
            opportunity_score += 0.1
        
        # Seasonal adjustments
        if indicators.quarter == 4 and not indicators.is_holiday_season:
            opportunity_score += 0.1  # Year-end budget spending
        elif indicators.is_tax_season:
            opportunity_score += 0.05  # Tax planning season
        
        return max(0.0, min(1.0, opportunity_score))
    
    def _calculate_risk_score(self, indicators: MarketIndicators, 
                            market_condition: MarketCondition) -> float:
        """Calculate market risk score"""
        
        risk_score = 0.3  # Base risk
        
        # Market condition risks
        condition_risks = {
            MarketCondition.RECESSION: 0.4,
            MarketCondition.BEAR_MARKET: 0.3,
            MarketCondition.VOLATILE: 0.2,
            MarketCondition.SIDEWAYS: 0.1,
            MarketCondition.RECOVERY: 0.0,
            MarketCondition.BULL_MARKET: -0.1
        }
        risk_score += condition_risks.get(market_condition, 0)
        
        # Economic risks
        if indicators.inflation_rate > 5:
            risk_score += 0.1
        if indicators.unemployment_rate > 6:
            risk_score += 0.1
        if indicators.interest_rates > 7:
            risk_score += 0.1
        
        # Market risks
        if indicators.volatility_index > 30:
            risk_score += 0.1
        
        # Industry risks
        if indicators.regulatory_changes > 0.7:
            risk_score += 0.1
        if indicators.market_saturation > 0.8:
            risk_score += 0.1
        if indicators.competitor_activity > 0.8:
            risk_score += 0.1
        
        return max(0.0, min(1.0, risk_score))
    
    def _generate_scoring_adjustments(self, market_condition: MarketCondition,
                                    seasonal_pattern: SeasonalPattern,
                                    trend_direction: TrendDirection,
                                    opportunity_score: float,
                                    risk_score: float) -> Dict[str, float]:
        """Generate recommended scoring adjustments"""
        
        adjustments = self.base_adjustments.copy()
        
        # Market condition adjustments
        condition_multipliers = {
            MarketCondition.BULL_MARKET: 1.15,
            MarketCondition.RECOVERY: 1.10,
            MarketCondition.SIDEWAYS: 1.00,
            MarketCondition.VOLATILE: 0.95,
            MarketCondition.BEAR_MARKET: 0.90,
            MarketCondition.RECESSION: 0.85
        }
        adjustments['lead_score_multiplier'] *= condition_multipliers.get(market_condition, 1.0)
        
        # Seasonal adjustments
        seasonal_adjustments = {
            SeasonalPattern.YEAR_END_RUSH: {'urgency_factor': 1.2, 'follow_up_frequency': 1.3},
            SeasonalPattern.Q1_STRONG: {'lead_score_multiplier': 1.1, 'conversion_probability': 1.1},
            SeasonalPattern.Q2_MODERATE: {'follow_up_frequency': 1.0},
            SeasonalPattern.Q3_SLOW: {'urgency_factor': 0.9, 'follow_up_frequency': 0.9},
            SeasonalPattern.HOLIDAY_IMPACT: {'urgency_factor': 0.8, 'follow_up_frequency': 0.7}
        }
        
        for key, value in seasonal_adjustments.get(seasonal_pattern, {}).items():
            adjustments[key] *= value
        
        # Trend direction adjustments
        if trend_direction == TrendDirection.UPWARD:
            adjustments['conversion_probability'] *= 1.1
        elif trend_direction == TrendDirection.DOWNWARD:
            adjustments['conversion_probability'] *= 0.9
            adjustments['urgency_factor'] *= 1.1  # More urgent follow-up needed
        
        # Opportunity and risk adjustments
        opportunity_factor = 0.8 + (opportunity_score * 0.4)  # Range: 0.8 - 1.2
        risk_factor = 1.2 - (risk_score * 0.4)  # Range: 0.8 - 1.2
        
        adjustments['lead_score_multiplier'] *= opportunity_factor * risk_factor
        adjustments['urgency_factor'] *= (2.0 - risk_factor)  # Higher risk = more urgency
        
        return adjustments
    
    def _calculate_confidence_level(self, indicators: MarketIndicators) -> float:
        """Calculate confidence level in the analysis"""
        
        confidence = 0.7  # Base confidence
        
        # Data completeness
        indicator_count = sum([
            1 for value in [
                indicators.gdp_growth, indicators.unemployment_rate,
                indicators.stock_market_performance, indicators.consumer_confidence
            ] if value != 0
        ])
        confidence += (indicator_count / 4.0) * 0.2
        
        # Market stability (lower volatility = higher confidence)
        volatility_penalty = indicators.volatility_index / 100.0
        confidence -= volatility_penalty
        
        return max(0.3, min(1.0, confidence))
    
    async def adapt_lead_scoring(self, lead_score: float, 
                               trend_analysis: TrendAnalysis) -> Dict[str, Any]:
        """Adapt lead scoring based on market trends"""
        
        try:
            adjustments = trend_analysis.recommended_adjustments
            
            # Apply adjustments
            adjusted_score = lead_score * adjustments['lead_score_multiplier']
            adjusted_urgency = adjustments['urgency_factor']
            adjusted_follow_up = adjustments['follow_up_frequency']
            adjusted_conversion = adjustments['conversion_probability']
            
            # Ensure score stays within bounds
            adjusted_score = max(0, min(100, adjusted_score))
            
            return {
                'original_score': lead_score,
                'adjusted_score': adjusted_score,
                'score_change': adjusted_score - lead_score,
                'urgency_multiplier': adjusted_urgency,
                'follow_up_frequency_multiplier': adjusted_follow_up,
                'conversion_probability_multiplier': adjusted_conversion,
                'market_condition': trend_analysis.market_condition.value,
                'seasonal_pattern': trend_analysis.seasonal_pattern.value,
                'confidence_level': trend_analysis.confidence_level,
                'adjustment_reason': self._generate_adjustment_reason(trend_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error adapting lead scoring: {e}")
            return {
                'original_score': lead_score,
                'adjusted_score': lead_score,
                'score_change': 0,
                'error': str(e)
            }
    
    def _generate_adjustment_reason(self, trend_analysis: TrendAnalysis) -> str:
        """Generate human-readable reason for adjustments"""
        
        reasons = []
        
        # Market condition reason
        if trend_analysis.market_condition == MarketCondition.BULL_MARKET:
            reasons.append("Strong market conditions favor higher conversion rates")
        elif trend_analysis.market_condition == MarketCondition.RECESSION:
            reasons.append("Economic downturn requires more conservative scoring")
        
        # Seasonal reason
        if trend_analysis.seasonal_pattern == SeasonalPattern.YEAR_END_RUSH:
            reasons.append("Year-end budget cycles create urgency")
        elif trend_analysis.seasonal_pattern == SeasonalPattern.HOLIDAY_IMPACT:
            reasons.append("Holiday season typically shows reduced activity")
        
        # Opportunity/risk reason
        if trend_analysis.opportunity_score > 0.7:
            reasons.append("High market opportunity detected")
        elif trend_analysis.risk_score > 0.7:
            reasons.append("Elevated market risks require careful approach")
        
        return "; ".join(reasons) if reasons else "Standard market conditions"

# Global market trend analysis engine
market_trend_engine = MarketTrendAnalysisEngine()