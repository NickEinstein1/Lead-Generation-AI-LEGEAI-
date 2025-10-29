from typing import Dict, List, Any, Optional, Union
import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import os
from enum import Enum

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    SOCIAL_MEDIA = "social_media"
    FINANCIAL = "financial"
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    MARKET = "market"
    COMPETITIVE = "competitive"

@dataclass
class DataSourceConfig:
    name: str
    api_endpoint: str
    api_key: str
    rate_limit: int  # requests per minute
    timeout: int = 30
    cache_ttl: int = 300  # 5 minutes
    enabled: bool = True

class RealTimeDataIntegrator:
    """Integrate real-time data sources for enhanced lead scoring"""
    
    def __init__(self, config_path: str = None):
        self.data_sources = self._load_data_sources(config_path)
        self.cache = {}
        self.rate_limiters = {}
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _load_data_sources(self, config_path: str) -> Dict[str, DataSourceConfig]:
        """Load data source configurations"""
        return {
            'clearbit': DataSourceConfig(
                name='Clearbit Enrichment',
                api_endpoint='https://person.clearbit.com/v2/combined/find',
                api_key=os.getenv('CLEARBIT_API_KEY', ''),
                rate_limit=600  # 600 requests per minute
            ),
            'fullcontact': DataSourceConfig(
                name='FullContact Person API',
                api_endpoint='https://api.fullcontact.com/v3/person.enrich',
                api_key=os.getenv('FULLCONTACT_API_KEY', ''),
                rate_limit=300
            ),
            'social_media': DataSourceConfig(
                name='Social Media Aggregator',
                api_endpoint='https://api.socialmedia-aggregator.com/v1/profile',
                api_key=os.getenv('SOCIAL_API_KEY', ''),
                rate_limit=1000
            ),
            'financial_data': DataSourceConfig(
                name='Financial Data Provider',
                api_endpoint='https://api.financial-provider.com/v1/consumer',
                api_key=os.getenv('FINANCIAL_API_KEY', ''),
                rate_limit=200
            ),
            'market_data': DataSourceConfig(
                name='Market Intelligence',
                api_endpoint='https://api.market-intel.com/v1/insurance',
                api_key=os.getenv('MARKET_API_KEY', ''),
                rate_limit=500
            )
        }
    
    async def enrich_lead_data(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich lead with real-time external data"""
        enriched = lead_data.copy()
        
        # Create enrichment tasks
        tasks = []
        
        # Social media enrichment
        if self._should_enrich('social_media', lead_data):
            tasks.append(self._get_social_signals(lead_data))
        
        # Financial data enrichment
        if self._should_enrich('financial', lead_data):
            tasks.append(self._get_financial_indicators(lead_data))
        
        # Demographic enrichment
        if self._should_enrich('demographic', lead_data):
            tasks.append(self._get_demographic_data(lead_data))
        
        # Market context
        tasks.append(self._get_market_conditions())
        
        # Competitive landscape
        tasks.append(self._get_competitor_rates(lead_data.get('location', 'US')))
        
        # Execute all enrichment tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Enrichment task {i} failed: {result}")
                    continue
                
                if result:
                    enriched.update(result)
            
            # Add enrichment metadata
            enriched['enrichment_metadata'] = {
                'timestamp': datetime.utcnow().isoformat(),
                'sources_used': [task.__name__ for task in tasks if not isinstance(result, Exception)],
                'enrichment_score': self._calculate_enrichment_score(enriched, lead_data)
            }
            
        except Exception as e:
            logger.error(f"Error during lead enrichment: {e}")
            enriched['enrichment_error'] = str(e)
        
        return enriched
    
    async def _get_social_signals(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract social media engagement signals"""
        email = lead_data.get('email')
        if not email:
            return {}
        
        cache_key = f"social_{hashlib.md5(email.encode()).hexdigest()}"
        
        # Check cache first
        if cached_data := self._get_cached_data(cache_key):
            return cached_data
        
        try:
            # Use FullContact API for social media data
            headers = {
                'Authorization': f"Bearer {self.data_sources['fullcontact'].api_key}",
                'Content-Type': 'application/json'
            }
            
            payload = {
                'email': email,
                'webhookUrl': None  # For real-time response
            }
            
            async with self.session.post(
                self.data_sources['fullcontact'].api_endpoint,
                headers=headers,
                json=payload,
                timeout=self.data_sources['fullcontact'].timeout
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    social_signals = {
                        'social_signals': {
                            'platforms': [],
                            'total_followers': 0,
                            'engagement_score': 0,
                            'influence_score': 0,
                            'professional_presence': False,
                            'social_activity_level': 'unknown'
                        }
                    }
                    
                    # Process social profiles
                    if 'socialProfiles' in data:
                        for profile in data['socialProfiles']:
                            platform_data = {
                                'platform': profile.get('type', '').lower(),
                                'followers': profile.get('followers', 0),
                                'following': profile.get('following', 0),
                                'url': profile.get('url', '')
                            }
                            social_signals['social_signals']['platforms'].append(platform_data)
                            social_signals['social_signals']['total_followers'] += platform_data['followers']
                            
                            # Check for professional presence
                            if platform_data['platform'] in ['linkedin', 'twitter']:
                                social_signals['social_signals']['professional_presence'] = True
                    
                    # Calculate engagement and influence scores
                    social_signals['social_signals']['engagement_score'] = min(10, 
                        social_signals['social_signals']['total_followers'] / 1000)
                    
                    social_signals['social_signals']['influence_score'] = self._calculate_influence_score(
                        social_signals['social_signals']['platforms'])
                    
                    # Determine activity level
                    if social_signals['social_signals']['total_followers'] > 5000:
                        social_signals['social_signals']['social_activity_level'] = 'high'
                    elif social_signals['social_signals']['total_followers'] > 500:
                        social_signals['social_signals']['social_activity_level'] = 'medium'
                    else:
                        social_signals['social_signals']['social_activity_level'] = 'low'
                    
                    # Cache the result
                    self._cache_data(cache_key, social_signals)
                    return social_signals
                
                else:
                    logger.warning(f"Social media API returned status {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error fetching social signals: {e}")
            return {}
    
    async def _get_financial_indicators(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get financial health indicators"""
        email = lead_data.get('email')
        location = lead_data.get('location', 'US')
        
        if not email:
            return {}
        
        cache_key = f"financial_{hashlib.md5(email.encode()).hexdigest()}"
        
        if cached_data := self._get_cached_data(cache_key):
            return cached_data
        
        try:
            # Simulate financial data API call (replace with actual provider)
            financial_indicators = {
                'financial_indicators': {
                    'credit_score_range': self._estimate_credit_score_range(lead_data),
                    'income_stability': self._assess_income_stability(lead_data),
                    'debt_to_income_estimate': self._estimate_debt_ratio(lead_data),
                    'financial_stress_indicators': self._detect_financial_stress(lead_data),
                    'insurance_affordability_score': 0,
                    'payment_behavior_score': 0
                }
            }
            
            # Calculate affordability score
            income = lead_data.get('income', 50000)
            age = lead_data.get('age', 35)
            
            affordability_factors = [
                min(10, income / 5000),  # Income factor
                max(0, 10 - abs(age - 40) / 5),  # Age factor (peak at 40)
                financial_indicators['financial_indicators']['income_stability'],
                10 - financial_indicators['financial_indicators']['debt_to_income_estimate']
            ]
            
            financial_indicators['financial_indicators']['insurance_affordability_score'] = \
                sum(affordability_factors) / len(affordability_factors)
            
            # Cache the result
            self._cache_data(cache_key, financial_indicators)
            return financial_indicators
        
        except Exception as e:
            logger.error(f"Error fetching financial indicators: {e}")
            return {}
    
    async def _get_demographic_data(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced demographic and lifestyle data"""
        email = lead_data.get('email')
        if not email:
            return {}
        
        cache_key = f"demo_{hashlib.md5(email.encode()).hexdigest()}"
        
        if cached_data := self._get_cached_data(cache_key):
            return cached_data
        
        try:
            # Use Clearbit for demographic enrichment
            headers = {
                'Authorization': f"Bearer {self.data_sources['clearbit'].api_key}"
            }
            
            params = {'email': email}
            
            async with self.session.get(
                self.data_sources['clearbit'].api_endpoint,
                headers=headers,
                params=params,
                timeout=self.data_sources['clearbit'].timeout
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    demographic_data = {
                        'demographic_enrichment': {
                            'employment': {},
                            'education': {},
                            'lifestyle': {},
                            'location_details': {},
                            'family_composition': {}
                        }
                    }
                    
                    # Process person data
                    if person := data.get('person'):
                        demographic_data['demographic_enrichment']['employment'] = {
                            'company': person.get('employment', {}).get('name', ''),
                            'title': person.get('employment', {}).get('title', ''),
                            'role': person.get('employment', {}).get('role', ''),
                            'seniority': person.get('employment', {}).get('seniority', ''),
                            'industry': person.get('employment', {}).get('domain', '')
                        }
                        
                        demographic_data['demographic_enrichment']['education'] = {
                            'level': 'unknown',  # Would be extracted from bio/description
                            'field': 'unknown'
                        }
                        
                        # Location details
                        if geo := person.get('geo'):
                            demographic_data['demographic_enrichment']['location_details'] = {
                                'city': geo.get('city', ''),
                                'state': geo.get('state', ''),
                                'country': geo.get('country', ''),
                                'timezone': geo.get('timezone', ''),
                                'metro_area': geo.get('metro', '')
                            }
                    
                    # Cache the result
                    self._cache_data(cache_key, demographic_data)
                    return demographic_data
                
                else:
                    logger.warning(f"Clearbit API returned status {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error fetching demographic data: {e}")
            return {}
    
    async def _get_market_conditions(self) -> Dict[str, Any]:
        """Get current insurance market conditions"""
        cache_key = "market_conditions"
        
        if cached_data := self._get_cached_data(cache_key, ttl=3600):  # 1 hour cache
            return cached_data
        
        try:
            # Simulate market data API (replace with actual provider)
            market_data = {
                'market_context': {
                    'insurance_market_trends': {
                        'auto_insurance': {
                            'avg_premium_change': 0.05,  # 5% increase
                            'market_competitiveness': 'high',
                            'rate_shopping_activity': 'elevated'
                        },
                        'home_insurance': {
                            'avg_premium_change': 0.08,  # 8% increase
                            'market_competitiveness': 'medium',
                            'rate_shopping_activity': 'normal'
                        },
                        'life_insurance': {
                            'avg_premium_change': 0.02,  # 2% increase
                            'market_competitiveness': 'high',
                            'rate_shopping_activity': 'high'
                        }
                    },
                    'economic_indicators': {
                        'unemployment_rate': 3.7,
                        'inflation_rate': 3.2,
                        'interest_rates': 5.25,
                        'consumer_confidence': 102.3
                    },
                    'seasonal_factors': {
                        'current_season': self._get_current_season(),
                        'seasonal_multiplier': self._get_seasonal_multiplier(),
                        'holiday_proximity': self._get_holiday_proximity()
                    }
                }
            }
            
            # Cache the result
            self._cache_data(cache_key, market_data, ttl=3600)
            return market_data
        
        except Exception as e:
            logger.error(f"Error fetching market conditions: {e}")
            return {}
    
    async def _get_competitor_rates(self, location: str) -> Dict[str, Any]:
        """Get competitive rate information for location"""
        cache_key = f"competitor_rates_{location}"
        
        if cached_data := self._get_cached_data(cache_key, ttl=1800):  # 30 min cache
            return cached_data
        
        try:
            # Simulate competitor rate API
            competitor_data = {
                'competitive_landscape': {
                    'auto_insurance': {
                        'market_avg_premium': 1200,
                        'our_competitive_position': 'below_average',  # 15% below market
                        'top_competitors': ['State Farm', 'GEICO', 'Progressive'],
                        'rate_advantage': 0.15
                    },
                    'home_insurance': {
                        'market_avg_premium': 1500,
                        'our_competitive_position': 'average',
                        'top_competitors': ['State Farm', 'Allstate', 'USAA'],
                        'rate_advantage': 0.05
                    },
                    'life_insurance': {
                        'market_avg_premium': 500,
                        'our_competitive_position': 'competitive',
                        'top_competitors': ['Northwestern Mutual', 'New York Life', 'MassMutual'],
                        'rate_advantage': 0.10
                    },
                    'location_factors': {
                        'risk_level': self._assess_location_risk(location),
                        'market_saturation': 'medium',
                        'regulatory_environment': 'standard'
                    }
                }
            }
            
            # Cache the result
            self._cache_data(cache_key, competitor_data, ttl=1800)
            return competitor_data
        
        except Exception as e:
            logger.error(f"Error fetching competitor rates: {e}")
            return {}
    
    # Helper methods
    def _should_enrich(self, data_type: str, lead_data: Dict[str, Any]) -> bool:
        """Determine if enrichment should be performed"""
        required_fields = {
            'social_media': ['email'],
            'financial': ['email', 'income'],
            'demographic': ['email']
        }
        
        if data_type not in required_fields:
            return True
        
        return all(field in lead_data and lead_data[field] for field in required_fields[data_type])
    
    def _get_cached_data(self, key: str, ttl: int = None) -> Optional[Dict[str, Any]]:
        """Get data from cache if not expired"""
        if key in self.cache:
            cached_item = self.cache[key]
            cache_ttl = ttl or 300  # Default 5 minutes
            
            if datetime.now() - cached_item['timestamp'] < timedelta(seconds=cache_ttl):
                return cached_item['data']
            else:
                del self.cache[key]
        
        return None
    
    def _cache_data(self, key: str, data: Dict[str, Any], ttl: int = None):
        """Cache data with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now(),
            'ttl': ttl or 300
        }
    
    def _calculate_enrichment_score(self, enriched_data: Dict[str, Any], 
                                  original_data: Dict[str, Any]) -> float:
        """Calculate how much the data was enriched"""
        original_fields = len([k for k, v in original_data.items() if v])
        enriched_fields = len([k for k, v in enriched_data.items() if v])
        
        return min(1.0, (enriched_fields - original_fields) / max(1, original_fields))
    
    def _calculate_influence_score(self, platforms: List[Dict[str, Any]]) -> float:
        """Calculate social influence score"""
        if not platforms:
            return 0
        
        total_score = 0
        platform_weights = {
            'linkedin': 3.0,
            'twitter': 2.5,
            'facebook': 2.0,
            'instagram': 1.5,
            'tiktok': 1.0
        }
        
        for platform in platforms:
            platform_name = platform.get('platform', '').lower()
            followers = platform.get('followers', 0)
            weight = platform_weights.get(platform_name, 1.0)
            
            # Logarithmic scale for followers
            if followers > 0:
                platform_score = min(10, (followers ** 0.3) * weight)
                total_score += platform_score
        
        return min(10, total_score / len(platforms))
    
    def _estimate_credit_score_range(self, lead_data: Dict[str, Any]) -> str:
        """Estimate credit score range based on available data"""
        income = lead_data.get('income', 50000)
        age = lead_data.get('age', 35)
        employment = lead_data.get('employment_status', 'employed')
        
        # Simple heuristic (replace with actual credit data)
        score = 650  # Base score
        
        if income > 75000:
            score += 50
        elif income > 50000:
            score += 25
        
        if age > 30:
            score += 20
        
        if employment == 'employed':
            score += 30
        elif employment == 'self_employed':
            score += 10
        
        if score >= 750:
            return 'excellent'
        elif score >= 700:
            return 'good'
        elif score >= 650:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_income_stability(self, lead_data: Dict[str, Any]) -> float:
        """Assess income stability (0-10 scale)"""
        employment = lead_data.get('employment_status', 'employed')
        age = lead_data.get('age', 35)
        
        stability_score = 5.0  # Base score
        
        if employment == 'employed':
            stability_score += 3.0
        elif employment == 'self_employed':
            stability_score += 1.0
        elif employment == 'retired':
            stability_score += 2.0
        
        # Age factor (more stable with age, up to a point)
        if 30 <= age <= 55:
            stability_score += 2.0
        elif 25 <= age < 30 or 55 < age <= 65:
            stability_score += 1.0
        
        return min(10.0, stability_score)
    
    def _estimate_debt_ratio(self, lead_data: Dict[str, Any]) -> float:
        """Estimate debt-to-income ratio (0-10 scale, higher = more debt)"""
        age = lead_data.get('age', 35)
        income = lead_data.get('income', 50000)
        
        # Heuristic based on age and income
        if age < 30:
            base_ratio = 6.0  # Higher debt for younger people
        elif age < 50:
            base_ratio = 4.0  # Moderate debt
        else:
            base_ratio = 3.0  # Lower debt for older people
        
        # Income adjustment
        if income > 100000:
            base_ratio -= 1.0
        elif income < 40000:
            base_ratio += 1.0
        
        return max(0, min(10, base_ratio))
    
    def _detect_financial_stress(self, lead_data: Dict[str, Any]) -> List[str]:
        """Detect potential financial stress indicators"""
        indicators = []
        
        income = lead_data.get('income', 50000)
        age = lead_data.get('age', 35)
        employment = lead_data.get('employment_status', 'employed')
        
        if income < 40000:
            indicators.append('low_income')
        
        if employment in ['unemployed', 'part_time']:
            indicators.append('employment_instability')
        
        if age < 25 and income < 35000:
            indicators.append('young_low_earner')
        
        return indicators
    
    def _get_current_season(self) -> str:
        """Get current season for seasonal adjustments"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _get_seasonal_multiplier(self) -> float:
        """Get seasonal multiplier for insurance shopping"""
        season = self._get_current_season()
        multipliers = {
            'spring': 1.2,  # High shopping season
            'summer': 1.1,  # Moderate shopping
            'fall': 1.3,    # Highest shopping (new year planning)
            'winter': 0.9   # Lower shopping
        }
        return multipliers.get(season, 1.0)
    
    def _get_holiday_proximity(self) -> Dict[str, Any]:
        """Check proximity to major holidays"""
        now = datetime.now()
        # Simplified holiday detection
        return {
            'days_to_next_major_holiday': 30,  # Placeholder
            'holiday_shopping_period': False,
            'end_of_year_planning': now.month == 12
        }
    
    def _assess_location_risk(self, location: str) -> str:
        """Assess risk level for location"""
        # Simplified location risk assessment
        high_risk_states = ['FL', 'CA', 'TX', 'LA']
        if any(state in location.upper() for state in high_risk_states):
            return 'high'
        return 'medium'

# Usage example
async def enrich_lead_example():
    """Example usage of real-time data integration"""
    
    sample_lead = {
        'lead_id': 'LEAD_12345',
        'email': 'john.doe@example.com',
        'age': 35,
        'income': 75000,
        'location': 'Austin, TX',
        'employment_status': 'employed'
    }
    
    async with RealTimeDataIntegrator() as integrator:
        enriched_lead = await integrator.enrich_lead_data(sample_lead)
        
        print("Enriched Lead Data:")
        print(json.dumps(enriched_lead, indent=2, default=str))
        
        return enriched_lead

if __name__ == "__main__":
    asyncio.run(enrich_lead_example())