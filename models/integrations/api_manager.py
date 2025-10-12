import asyncio
import aiohttp
import time
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class IntegrationStatus(Enum):
    """Integration status types"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"

class DataSourceType(Enum):
    """Types of data sources"""
    ENRICHMENT = "enrichment"
    SOCIAL_MEDIA = "social_media"
    FINANCIAL = "financial"
    MARKET_INTELLIGENCE = "market_intelligence"
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    COMPLIANCE = "compliance"

@dataclass
class APICredentials:
    """API credentials configuration"""
    api_key: str
    secret_key: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if credentials are expired"""
        if not self.expires_at:
            return False
        return datetime.now() >= self.expires_at

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    backoff_factor: float = 2.0
    max_retries: int = 3

@dataclass
class IntegrationConfig:
    """Integration configuration"""
    name: str
    source_type: DataSourceType
    base_url: str
    credentials: APICredentials
    rate_limits: RateLimitConfig
    timeout: int = 30
    retry_config: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    health_check_endpoint: Optional[str] = None
    
class APIResponse:
    """Standardized API response"""
    
    def __init__(self, status_code: int, data: Any, headers: Dict[str, str] = None,
                 response_time: float = 0, error: str = None):
        self.status_code = status_code
        self.data = data
        self.headers = headers or {}
        self.response_time = response_time
        self.error = error
        self.timestamp = datetime.now()
        self.success = 200 <= status_code < 300

class BaseIntegration(ABC):
    """Base class for all integrations"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.status = IntegrationStatus.INACTIVE
        self.last_request_time = 0
        self.request_count = defaultdict(int)
        self.error_count = 0
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.rate_limiter = RateLimiter(config.rate_limits)
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'last_success': None,
            'last_error': None
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the integration"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.config.headers
            )
        
        # Test connection
        if await self.health_check():
            self.status = IntegrationStatus.ACTIVE
            logger.info(f"Integration {self.config.name} initialized successfully")
        else:
            self.status = IntegrationStatus.ERROR
            logger.error(f"Integration {self.config.name} failed health check")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def health_check(self) -> bool:
        """Perform health check"""
        if not self.config.health_check_endpoint:
            return True
        
        try:
            response = await self.make_request(
                method="GET",
                endpoint=self.config.health_check_endpoint,
                timeout=10
            )
            return response.success
        except Exception as e:
            logger.error(f"Health check failed for {self.config.name}: {e}")
            return False
    
    async def make_request(self, method: str, endpoint: str, 
                          data: Dict[str, Any] = None, 
                          params: Dict[str, Any] = None,
                          timeout: int = None) -> APIResponse:
        """Make HTTP request with rate limiting and error handling"""
        
        # Check rate limits
        if not await self.rate_limiter.acquire():
            raise Exception("Rate limit exceeded")
        
        # Prepare request
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = self._get_auth_headers()
        timeout = timeout or self.config.timeout
        
        start_time = time.time()
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                response_time = time.time() - start_time
                response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                
                api_response = APIResponse(
                    status_code=response.status,
                    data=response_data,
                    headers=dict(response.headers),
                    response_time=response_time
                )
                
                # Update metrics
                self._update_metrics(api_response)
                
                return api_response
                
        except Exception as e:
            response_time = time.time() - start_time
            error_response = APIResponse(
                status_code=500,
                data=None,
                response_time=response_time,
                error=str(e)
            )
            
            self._update_metrics(error_response)
            raise
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        headers = {}
        
        if self.config.credentials.api_key:
            headers['Authorization'] = f"Bearer {self.config.credentials.api_key}"
        
        if self.config.credentials.access_token:
            headers['Authorization'] = f"Bearer {self.config.credentials.access_token}"
        
        return headers
    
    def _update_metrics(self, response: APIResponse):
        """Update integration metrics"""
        self.metrics['total_requests'] += 1
        
        if response.success:
            self.metrics['successful_requests'] += 1
            self.metrics['last_success'] = response.timestamp
        else:
            self.metrics['failed_requests'] += 1
            self.metrics['last_error'] = response.timestamp
            self.error_count += 1
        
        # Update average response time
        total_time = (self.metrics['avg_response_time'] * (self.metrics['total_requests'] - 1) + 
                     response.response_time)
        self.metrics['avg_response_time'] = total_time / self.metrics['total_requests']
    
    @abstractmethod
    async def enrich_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data using this integration"""
        pass

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests = deque()
        self.minute_requests = deque()
        self.hour_requests = deque()
        self.day_requests = deque()
    
    async def acquire(self) -> bool:
        """Acquire permission to make a request"""
        now = time.time()
        
        # Clean old requests
        self._clean_old_requests(now)
        
        # Check limits
        if (len(self.minute_requests) >= self.config.requests_per_minute or
            len(self.hour_requests) >= self.config.requests_per_hour or
            len(self.day_requests) >= self.config.requests_per_day):
            return False
        
        # Record request
        self.minute_requests.append(now)
        self.hour_requests.append(now)
        self.day_requests.append(now)
        
        return True
    
    def _clean_old_requests(self, now: float):
        """Remove old requests from tracking"""
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400
        
        while self.minute_requests and self.minute_requests[0] < minute_ago:
            self.minute_requests.popleft()
        
        while self.hour_requests and self.hour_requests[0] < hour_ago:
            self.hour_requests.popleft()
        
        while self.day_requests and self.day_requests[0] < day_ago:
            self.day_requests.popleft()

class ClearbitIntegration(BaseIntegration):
    """Clearbit API integration for person/company enrichment"""
    
    async def enrich_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich lead data using Clearbit"""
        email = input_data.get('email')
        if not email:
            return {}
        
        try:
            response = await self.make_request(
                method="GET",
                endpoint="combined/find",
                params={'email': email}
            )
            
            if response.success:
                return self._parse_clearbit_response(response.data)
            else:
                logger.warning(f"Clearbit API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Clearbit enrichment failed: {e}")
            return {}
    
    def _parse_clearbit_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Clearbit API response"""
        enriched = {}
        
        if 'person' in data:
            person = data['person']
            enriched.update({
                'full_name': person.get('name', {}).get('fullName'),
                'location': person.get('location'),
                'employment': person.get('employment', {}),
                'social_profiles': person.get('socialProfiles', [])
            })
        
        if 'company' in data:
            company = data['company']
            enriched.update({
                'company_name': company.get('name'),
                'company_industry': company.get('category', {}).get('industry'),
                'company_size': company.get('metrics', {}).get('employees'),
                'company_revenue': company.get('metrics', {}).get('annualRevenue')
            })
        
        return enriched

class FullContactIntegration(BaseIntegration):
    """FullContact API integration for social media data"""
    
    async def enrich_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich lead data using FullContact"""
        email = input_data.get('email')
        if not email:
            return {}
        
        try:
            response = await self.make_request(
                method="POST",
                endpoint="person.enrich",
                data={'email': email}
            )
            
            if response.success:
                return self._parse_fullcontact_response(response.data)
            else:
                logger.warning(f"FullContact API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"FullContact enrichment failed: {e}")
            return {}
    
    def _parse_fullcontact_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FullContact API response"""
        enriched = {}
        
        if 'socialProfiles' in data:
            social_profiles = {}
            for profile in data['socialProfiles']:
                platform = profile.get('type', '').lower()
                social_profiles[platform] = {
                    'url': profile.get('url'),
                    'username': profile.get('username'),
                    'followers': profile.get('followers', 0)
                }
            enriched['social_profiles'] = social_profiles
        
        if 'demographics' in data:
            demographics = data['demographics']
            enriched.update({
                'age_range': demographics.get('ageRange'),
                'gender': demographics.get('gender'),
                'location_general': demographics.get('locationGeneral')
            })
        
        return enriched

class SocialMediaIntegration(BaseIntegration):
    """Social media aggregator integration"""
    
    async def enrich_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich lead data with social media insights"""
        email = input_data.get('email')
        if not email:
            return {}
        
        try:
            response = await self.make_request(
                method="GET",
                endpoint="profile",
                params={'email': email, 'include': 'engagement,interests,activity'}
            )
            
            if response.success:
                return self._parse_social_response(response.data)
            else:
                logger.warning(f"Social Media API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Social media enrichment failed: {e}")
            return {}
    
    def _parse_social_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse social media API response"""
        return {
            'social_engagement_score': data.get('engagementScore', 0),
            'interests': data.get('interests', []),
            'activity_level': data.get('activityLevel', 'unknown'),
            'influence_score': data.get('influenceScore', 0),
            'platform_presence': data.get('platformPresence', {})
        }

class FinancialDataIntegration(BaseIntegration):
    """Financial data provider integration"""
    
    async def enrich_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich lead data with financial insights"""
        # Use email hash for privacy
        email = input_data.get('email')
        if not email:
            return {}
        
        email_hash = hashlib.sha256(email.encode()).hexdigest()
        
        try:
            response = await self.make_request(
                method="POST",
                endpoint="consumer/insights",
                data={
                    'identifier': email_hash,
                    'include': ['creditScore', 'income', 'assets', 'spending']
                }
            )
            
            if response.success:
                return self._parse_financial_response(response.data)
            else:
                logger.warning(f"Financial Data API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Financial data enrichment failed: {e}")
            return {}
    
    def _parse_financial_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse financial data API response"""
        return {
            'estimated_income_range': data.get('incomeRange'),
            'credit_score_range': data.get('creditScoreRange'),
            'spending_patterns': data.get('spendingPatterns', {}),
            'financial_stability_score': data.get('stabilityScore', 0),
            'insurance_likelihood': data.get('insuranceLikelihood', 0)
        }

class MarketIntelligenceIntegration(BaseIntegration):
    """Market intelligence integration"""
    
    async def enrich_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich lead data with market intelligence"""
        location = input_data.get('location') or input_data.get('zip_code')
        if not location:
            return {}
        
        try:
            response = await self.make_request(
                method="GET",
                endpoint="insurance/market-data",
                params={
                    'location': location,
                    'include': 'competition,pricing,demand'
                }
            )
            
            if response.success:
                return self._parse_market_response(response.data)
            else:
                logger.warning(f"Market Intelligence API error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Market intelligence enrichment failed: {e}")
            return {}
    
    def _parse_market_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse market intelligence API response"""
        return {
            'market_competition_level': data.get('competitionLevel', 'medium'),
            'average_premium_range': data.get('averagePremiumRange'),
            'market_demand_score': data.get('demandScore', 0),
            'seasonal_trends': data.get('seasonalTrends', {}),
            'local_regulations': data.get('regulations', [])
        }

class IntegrationManager:
    """Manages all external API integrations"""
    
    def __init__(self, config_file: str = None):
        self.integrations: Dict[str, BaseIntegration] = {}
        self.config = self._load_config(config_file)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize integrations
        self._initialize_integrations()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load integration configurations"""
        # Default configuration
        default_config = {
            'clearbit': {
                'enabled': True,
                'api_key': os.getenv('CLEARBIT_API_KEY', ''),
                'base_url': 'https://person.clearbit.com/v2',
                'rate_limit': {'requests_per_minute': 600}
            },
            'fullcontact': {
                'enabled': True,
                'api_key': os.getenv('FULLCONTACT_API_KEY', ''),
                'base_url': 'https://api.fullcontact.com/v3',
                'rate_limit': {'requests_per_minute': 300}
            },
            'social_media': {
                'enabled': True,
                'api_key': os.getenv('SOCIAL_API_KEY', ''),
                'base_url': 'https://api.socialmedia-aggregator.com/v1',
                'rate_limit': {'requests_per_minute': 1000}
            },
            'financial_data': {
                'enabled': True,
                'api_key': os.getenv('FINANCIAL_API_KEY', ''),
                'base_url': 'https://api.financial-provider.com/v1',
                'rate_limit': {'requests_per_minute': 200}
            },
            'market_intelligence': {
                'enabled': True,
                'api_key': os.getenv('MARKET_API_KEY', ''),
                'base_url': 'https://api.market-intel.com/v1',
                'rate_limit': {'requests_per_minute': 500}
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                default_config.update(file_config)
        
        return default_config
    
    def _initialize_integrations(self):
        """Initialize all configured integrations"""
        integration_classes = {
            'clearbit': ClearbitIntegration,
            'fullcontact': FullContactIntegration,
            'social_media': SocialMediaIntegration,
            'financial_data': FinancialDataIntegration,
            'market_intelligence': MarketIntelligenceIntegration
        }
        
        for name, config in self.config.items():
            if config.get('enabled', False) and config.get('api_key'):
                integration_class = integration_classes.get(name)
                if integration_class:
                    # Create integration config
                    integration_config = IntegrationConfig(
                        name=name,
                        source_type=DataSourceType.ENRICHMENT,
                        base_url=config['base_url'],
                        credentials=APICredentials(api_key=config['api_key']),
                        rate_limits=RateLimitConfig(**config.get('rate_limit', {})),
                        timeout=config.get('timeout', 30),
                        enabled=True
                    )
                    
                    # Create integration instance
                    self.integrations[name] = integration_class(integration_config)
                    logger.info(f"Initialized {name} integration")
    
    async def enrich_lead_data(self, lead_data: Dict[str, Any], 
                              sources: List[str] = None) -> Dict[str, Any]:
        """Enrich lead data using specified or all available sources"""
        if sources is None:
            sources = list(self.integrations.keys())
        
        # Check cache first
        cache_key = self._generate_cache_key(lead_data, sources)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        enriched_data = lead_data.copy()
        enrichment_results = {}
        
        # Run enrichments in parallel
        tasks = []
        for source_name in sources:
            if source_name in self.integrations:
                integration = self.integrations[source_name]
                if integration.status == IntegrationStatus.ACTIVE:
                    tasks.append(self._enrich_with_source(integration, lead_data))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                source_name = sources[i] if i < len(sources) else f"source_{i}"
                if isinstance(result, Exception):
                    logger.error(f"Enrichment failed for {source_name}: {result}")
                    enrichment_results[source_name] = {'error': str(result)}
                else:
                    enrichment_results[source_name] = result
                    enriched_data.update(result)
        
        # Add enrichment metadata
        enriched_data['_enrichment'] = {
            'sources_used': sources,
            'results': enrichment_results,
            'timestamp': datetime.now().isoformat(),
            'cache_key': cache_key
        }
        
        # Cache the result
        self.cache[cache_key] = (enriched_data, time.time())
        
        return enriched_data
    
    async def _enrich_with_source(self, integration: BaseIntegration, 
                                 lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data with a specific source"""
        async with integration:
            return await integration.enrich_data(lead_data)
    
    def _generate_cache_key(self, lead_data: Dict[str, Any], 
                           sources: List[str]) -> str:
        """Generate cache key for lead data and sources"""
        # Use email as primary identifier
        email = lead_data.get('email', '')
        sources_str = ','.join(sorted(sources))
        
        key_data = f"{email}:{sources_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        status = {}
        
        for name, integration in self.integrations.items():
            health_ok = await integration.health_check()
            
            status[name] = {
                'status': integration.status.value,
                'health_check': health_ok,
                'metrics': integration.metrics,
                'error_count': integration.error_count,
                'rate_limit_status': {
                    'minute_requests': len(integration.rate_limiter.minute_requests),
                    'hour_requests': len(integration.rate_limiter.hour_requests),
                    'day_requests': len(integration.rate_limiter.day_requests)
                }
            }
        
        return status
    
    async def test_integration(self, integration_name: str, 
                              test_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test a specific integration"""
        if integration_name not in self.integrations:
            return {'error': f'Integration {integration_name} not found'}
        
        integration = self.integrations[integration_name]
        test_data = test_data or {'email': 'test@example.com'}
        
        try:
            async with integration:
                result = await integration.enrich_data(test_data)
                return {
                    'success': True,
                    'result': result,
                    'response_time': integration.metrics['avg_response_time']
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_cache(self):
        """Clear the enrichment cache"""
        self.cache.clear()
        logger.info("Integration cache cleared")

# Global integration manager instance
integration_manager = IntegrationManager()

# Usage example
async def example_usage():
    """Example of using the integration manager"""
    
    sample_lead = {
        'email': 'john.doe@example.com',
        'age': 35,
        'location': 'Austin, TX'
    }
    
    # Enrich with all available sources
    enriched_lead = await integration_manager.enrich_lead_data(sample_lead)
    
    print("Enriched Lead Data:")
    print(json.dumps(enriched_lead, indent=2, default=str))
    
    # Get integration status
    status = await integration_manager.get_integration_status()
    print("\nIntegration Status:")
    print(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(example_usage())