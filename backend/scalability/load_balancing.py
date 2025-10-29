import asyncio
import aiohttp
import redis
import json
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import random
import statistics
from collections import defaultdict, deque
import pickle

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    GEOGRAPHIC = "geographic"
    HEALTH_AWARE = "health_aware"

class ServerStatus(Enum):
    """Server status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"

@dataclass
class Server:
    """Server instance definition"""
    id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 1000
    current_connections: int = 0
    status: ServerStatus = ServerStatus.HEALTHY
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    total_requests: int = 0
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests
    
    @property
    def load_factor(self) -> float:
        """Calculate server load factor (0.0 to 1.0+)"""
        connection_load = self.current_connections / self.max_connections
        response_time_load = min(self.avg_response_time / 1.0, 1.0)  # Normalize to 1 second
        error_load = min(self.error_rate * 10, 1.0)  # 10% error rate = full load
        
        return (connection_load + response_time_load + error_load) / 3

class LoadBalancer:
    """Intelligent load balancer with multiple strategies"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_AWARE):
        self.strategy = strategy
        self.servers: Dict[str, Server] = {}
        self.current_index = 0
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Health checking
        self.health_check_interval = 30
        self.health_check_timeout = 5
        self.health_check_path = "/health"
        
        # Circuit breaker
        self.circuit_breaker_threshold = 5  # Failures before opening circuit
        self.circuit_breaker_timeout = 60   # Seconds before trying again
        self.circuit_breaker_state: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.request_stats = defaultdict(int)
        self.response_time_history = deque(maxlen=1000)
        
    async def __aenter__(self):
        """Async context manager entry"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def add_server(self, server: Server):
        """Add a server to the load balancer"""
        self.servers[server.id] = server
        self.circuit_breaker_state[server.id] = {
            'state': 'closed',  # closed, open, half-open
            'failure_count': 0,
            'last_failure': None,
            'next_attempt': None
        }
        logger.info(f"Added server {server.id} ({server.url}) to load balancer")
    
    def remove_server(self, server_id: str):
        """Remove a server from the load balancer"""
        if server_id in self.servers:
            del self.servers[server_id]
            del self.circuit_breaker_state[server_id]
            logger.info(f"Removed server {server_id} from load balancer")
    
    def get_healthy_servers(self) -> List[Server]:
        """Get list of healthy servers"""
        healthy_servers = []
        
        for server in self.servers.values():
            # Check server status
            if server.status != ServerStatus.HEALTHY:
                continue
            
            # Check circuit breaker
            cb_state = self.circuit_breaker_state[server.id]
            if cb_state['state'] == 'open':
                # Check if we should try again
                if (cb_state['next_attempt'] and 
                    datetime.now() >= cb_state['next_attempt']):
                    cb_state['state'] = 'half-open'
                    healthy_servers.append(server)
                continue
            
            healthy_servers.append(server)
        
        return healthy_servers
    
    def select_server(self, request_context: Dict[str, Any] = None) -> Optional[Server]:
        """Select a server based on the load balancing strategy"""
        healthy_servers = self.get_healthy_servers()
        
        if not healthy_servers:
            logger.warning("No healthy servers available")
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash_selection(healthy_servers, request_context)
        elif self.strategy == LoadBalancingStrategy.HEALTH_AWARE:
            return self._health_aware_selection(healthy_servers)
        else:
            return healthy_servers[0]  # Fallback
    
    def _round_robin_selection(self, servers: List[Server]) -> Server:
        """Round-robin server selection"""
        server = servers[self.current_index % len(servers)]
        self.current_index += 1
        return server
    
    def _weighted_round_robin_selection(self, servers: List[Server]) -> Server:
        """Weighted round-robin selection"""
        total_weight = sum(server.weight for server in servers)
        if total_weight == 0:
            return servers[0]
        
        # Create weighted list
        weighted_servers = []
        for server in servers:
            weighted_servers.extend([server] * server.weight)
        
        server = weighted_servers[self.current_index % len(weighted_servers)]
        self.current_index += 1
        return server
    
    def _least_connections_selection(self, servers: List[Server]) -> Server:
        """Select server with least connections"""
        return min(servers, key=lambda s: s.current_connections)
    
    def _least_response_time_selection(self, servers: List[Server]) -> Server:
        """Select server with lowest response time"""
        return min(servers, key=lambda s: s.avg_response_time)
    
    def _ip_hash_selection(self, servers: List[Server], 
                          request_context: Dict[str, Any]) -> Server:
        """Select server based on client IP hash"""
        client_ip = request_context.get('client_ip', '127.0.0.1')
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return servers[hash_value % len(servers)]
    
    def _health_aware_selection(self, servers: List[Server]) -> Server:
        """Select server based on health metrics"""
        # Calculate health scores
        server_scores = []
        for server in servers:
            # Lower load factor = higher score
            load_score = 1.0 - min(server.load_factor, 1.0)
            
            # Lower error rate = higher score
            error_score = 1.0 - min(server.error_rate, 1.0)
            
            # Faster response time = higher score
            response_score = 1.0 - min(server.avg_response_time / 2.0, 1.0)
            
            # Weight factors
            total_score = (load_score * 0.4 + error_score * 0.3 + 
                          response_score * 0.3) * server.weight
            
            server_scores.append((server, total_score))
        
        # Select server with highest score
        return max(server_scores, key=lambda x: x[1])[0]
    
    async def forward_request(self, method: str, path: str, 
                            request_context: Dict[str, Any] = None,
                            **kwargs) -> Dict[str, Any]:
        """Forward request to selected server"""
        server = self.select_server(request_context or {})
        
        if not server:
            raise Exception("No healthy servers available")
        
        # Track connection
        server.current_connections += 1
        server.total_requests += 1
        
        start_time = time.time()
        
        try:
            url = f"{server.url}{path}"
            
            async with self.session.request(method, url, **kwargs) as response:
                response_time = time.time() - start_time
                
                # Record metrics
                server.response_times.append(response_time)
                self.response_time_history.append(response_time)
                
                # Update circuit breaker
                self._update_circuit_breaker(server.id, success=True)
                
                # Return response data
                content = await response.text()
                
                return {
                    'status_code': response.status,
                    'content': content,
                    'headers': dict(response.headers),
                    'server_id': server.id,
                    'response_time': response_time
                }
                
        except Exception as e:
            # Record error
            server.error_count += 1
            response_time = time.time() - start_time
            
            # Update circuit breaker
            self._update_circuit_breaker(server.id, success=False)
            
            logger.error(f"Request failed on server {server.id}: {e}")
            raise
            
        finally:
            server.current_connections -= 1
    
    def _update_circuit_breaker(self, server_id: str, success: bool):
        """Update circuit breaker state"""
        cb_state = self.circuit_breaker_state[server_id]
        
        if success:
            if cb_state['state'] == 'half-open':
                # Reset circuit breaker
                cb_state['state'] = 'closed'
                cb_state['failure_count'] = 0
                logger.info(f"Circuit breaker closed for server {server_id}")
        else:
            cb_state['failure_count'] += 1
            cb_state['last_failure'] = datetime.now()
            
            if (cb_state['state'] == 'closed' and 
                cb_state['failure_count'] >= self.circuit_breaker_threshold):
                # Open circuit breaker
                cb_state['state'] = 'open'
                cb_state['next_attempt'] = (
                    datetime.now() + timedelta(seconds=self.circuit_breaker_timeout)
                )
                logger.warning(f"Circuit breaker opened for server {server_id}")
    
    async def start_health_checks(self):
        """Start periodic health checks"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self):
        """Perform health checks on all servers"""
        if not self.session:
            return
        
        for server in self.servers.values():
            try:
                start_time = time.time()
                url = f"{server.url}{self.health_check_path}"
                
                async with self.session.get(url, timeout=self.health_check_timeout) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        if server.status != ServerStatus.HEALTHY:
                            server.status = ServerStatus.HEALTHY
                            logger.info(f"Server {server.id} is now healthy")
                    else:
                        server.status = ServerStatus.UNHEALTHY
                        logger.warning(f"Server {server.id} health check failed: {response.status}")
                    
                    server.last_health_check = datetime.now()
                    
            except Exception as e:
                server.status = ServerStatus.UNHEALTHY
                server.last_health_check = datetime.now()
                logger.warning(f"Server {server.id} health check failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        healthy_count = len(self.get_healthy_servers())
        total_count = len(self.servers)
        
        total_requests = sum(server.total_requests for server in self.servers.values())
        total_errors = sum(server.error_count for server in self.servers.values())
        
        avg_response_time = (
            statistics.mean(self.response_time_history) 
            if self.response_time_history else 0.0
        )
        
        return {
            'strategy': self.strategy.value,
            'total_servers': total_count,
            'healthy_servers': healthy_count,
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': total_errors / total_requests if total_requests > 0 else 0.0,
            'avg_response_time': avg_response_time,
            'server_details': {
                server.id: {
                    'status': server.status.value,
                    'connections': server.current_connections,
                    'requests': server.total_requests,
                    'errors': server.error_count,
                    'avg_response_time': server.avg_response_time,
                    'load_factor': server.load_factor
                }
                for server in self.servers.values()
            }
        }

class DistributedCache:
    """Distributed caching system with Redis backend"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 2,
            'decode_responses': False
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.max_key_length = 250
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        self.cache_deletes = 0
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create deterministic key from arguments
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (dict, list)):
                key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest()[:8])
            else:
                key_parts.append(str(arg))
        
        # Add keyword arguments
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
            key_parts.append(hashlib.md5(kwargs_str.encode()).hexdigest()[:8])
        
        cache_key = ":".join(key_parts)
        
        # Truncate if too long
        if len(cache_key) > self.max_key_length:
            cache_key = cache_key[:self.max_key_length-8] + hashlib.md5(cache_key.encode()).hexdigest()[:8]
        
        return cache_key
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            cached_data = self.redis_client.get(key)
            
            if cached_data is not None:
                self.cache_hits += 1
                return pickle.loads(cached_data)
            else:
                self.cache_misses += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.cache_misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            serialized_value = pickle.dumps(value)
            
            result = self.redis_client.setex(key, ttl, serialized_value)
            
            if result:
                self.cache_sets += 1
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            result = self.redis_client.delete(key)
            
            if result:
                self.cache_deletes += 1
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                self.cache_deletes += deleted
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    def cache_decorator(self, prefix: str = "cache", ttl: int = None):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(prefix, func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_operations = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_operations if total_operations > 0 else 0.0
        
        # Get Redis info
        try:
            redis_info = self.redis_client.info()
            memory_usage = redis_info.get('used_memory_human', 'Unknown')
            connected_clients = redis_info.get('connected_clients', 0)
            total_keys = self.redis_client.dbsize()
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            memory_usage = 'Unknown'
            connected_clients = 0
            total_keys = 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_sets': self.cache_sets,
            'cache_deletes': self.cache_deletes,
            'hit_rate': hit_rate,
            'total_keys': total_keys,
            'memory_usage': memory_usage,
            'connected_clients': connected_clients
        }

class CachedLeadScorer:
    """Lead scorer with intelligent caching"""
    
    def __init__(self, cache: DistributedCache):
        self.cache = cache
        from backend.models.meta_lead_generation.inference import MetaLeadGenerationInference
        self.scorer = MetaLeadGenerationInference()
    
    @property
    def cache_decorator(self):
        return self.cache.cache_decorator("lead_scoring", ttl=1800)  # 30 minutes
    
    async def score_lead_cached(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score lead with caching"""
        # Generate cache key based on lead data
        cache_key = self.cache._generate_cache_key("lead_score", lead_data)
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for lead scoring: {cache_key}")
            return cached_result
        
        # Score the lead
        result = self.scorer.score_lead(lead_data)
        
        # Cache the result
        await self.cache.set(cache_key, result, ttl=1800)
        logger.debug(f"Cached lead scoring result: {cache_key}")
        
        return result
    
    async def batch_score_cached(self, leads_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch score leads with caching"""
        results = []
        cache_misses = []
        
        # Check cache for each lead
        for i, lead_data in enumerate(leads_data):
            cache_key = self.cache._generate_cache_key("lead_score", lead_data)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result is not None:
                results.append((i, cached_result))
            else:
                cache_misses.append((i, lead_data))
        
        # Score cache misses
        if cache_misses:
            miss_data = [lead_data for _, lead_data in cache_misses]
            miss_results = self.scorer.batch_score(miss_data)
            
            # Cache and add results
            for (i, lead_data), result in zip(cache_misses, miss_results):
                cache_key = self.cache._generate_cache_key("lead_score", lead_data)
                await self.cache.set(cache_key, result, ttl=1800)
                results.append((i, result))
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

# Global instances
distributed_cache = DistributedCache()
cached_lead_scorer = CachedLeadScorer(distributed_cache)

# Load balancer for API services
api_load_balancer = LoadBalancer(LoadBalancingStrategy.HEALTH_AWARE)

# Example usage
async def setup_load_balancer():
    """Setup load balancer with API servers"""
    
    # Add API servers
    servers = [
        Server("api-1", "localhost", 8000, weight=2),
        Server("api-2", "localhost", 8001, weight=2),
        Server("api-3", "localhost", 8002, weight=1),
    ]
    
    for server in servers:
        api_load_balancer.add_server(server)
    
    # Start health checks
    health_check_task = asyncio.create_task(api_load_balancer.start_health_checks())
    
    return health_check_task

async def example_usage():
    """Example of using load balancer and cache"""
    
    # Setup load balancer
    health_task = await setup_load_balancer()
    
    try:
        async with api_load_balancer:
            # Test load balancing
            for i in range(10):
                try:
                    response = await api_load_balancer.forward_request(
                        "GET", "/health",
                        request_context={"client_ip": f"192.168.1.{i}"}
                    )
                    print(f"Request {i}: Server {response['server_id']}, "
                          f"Response time: {response['response_time']:.3f}s")
                except Exception as e:
                    print(f"Request {i} failed: {e}")
                
                await asyncio.sleep(0.1)
            
            # Test caching
            lead_data = {
                'email': 'test@example.com',
                'age': 30,
                'income': 50000
            }
            
            # First call (cache miss)
            start_time = time.time()
            result1 = await cached_lead_scorer.score_lead_cached(lead_data)
            time1 = time.time() - start_time
            
            # Second call (cache hit)
            start_time = time.time()
            result2 = await cached_lead_scorer.score_lead_cached(lead_data)
            time2 = time.time() - start_time
            
            print(f"First call (cache miss): {time1:.3f}s")
            print(f"Second call (cache hit): {time2:.3f}s")
            print(f"Speedup: {time1/time2:.1f}x")
            
            # Get statistics
            lb_stats = api_load_balancer.get_stats()
            cache_stats = distributed_cache.get_stats()
            
            print("Load Balancer Stats:", json.dumps(lb_stats, indent=2, default=str))
            print("Cache Stats:", json.dumps(cache_stats, indent=2, default=str))
    
    finally:
        health_task.cancel()

if __name__ == "__main__":
    asyncio.run(example_usage())

