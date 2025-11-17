"""
Advanced Caching Strategy for Insurance Lead Scoring Platform

Provides multi-level caching, cache warming, invalidation strategies,
and intelligent cache management for optimal performance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
import hashlib
import aioredis
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"

class CacheStrategy(Enum):
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    CACHE_ASIDE = "cache_aside"

class EvictionPolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"

@dataclass
class CacheEntry:
    key: str
    value: Any
    ttl: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    writes: int = 0
    deletes: int = 0
    memory_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class MultiLevelCache:
    """Advanced multi-level caching system"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        # L1 Cache (Memory)
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l1_max_size = 10000
        self.l1_max_memory = 100 * 1024 * 1024  # 100MB
        
        # L2 Cache (Redis)
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 4
        }
        self.redis_pool = None
        
        # Cache configuration
        self.default_ttl = 3600
        self.eviction_policy = EvictionPolicy.LRU
        self.cache_strategy = CacheStrategy.CACHE_ASIDE
        
        # Statistics
        self.l1_stats = CacheStats()
        self.l2_stats = CacheStats()
        
        # Cache warming
        self.warming_tasks = {}
        self.warming_enabled = True
        
    async def initialize(self):
        """Initialize cache system"""
        
        # Initialize Redis connection
        self.redis_pool = aioredis.ConnectionPool.from_url(
            f"redis://{self.redis_config['host']}:{self.redis_config['port']}/{self.redis_config['db']}",
            max_connections=20
        )
        
        # Start background tasks
        asyncio.create_task(self._cache_maintenance_loop())
        asyncio.create_task(self._cache_warming_loop())
        
        logger.info("Multi-level cache system initialized")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with multi-level lookup"""
        
        # Try L1 cache first
        l1_result = await self._get_from_l1(key)
        if l1_result is not None:
            self.l1_stats.hits += 1
            return l1_result
        
        self.l1_stats.misses += 1
        
        # Try L2 cache (Redis)
        l2_result = await self._get_from_l2(key)
        if l2_result is not None:
            self.l2_stats.hits += 1
            # Promote to L1
            await self._set_to_l1(key, l2_result, self.default_ttl)
            return l2_result
        
        self.l2_stats.misses += 1
        return default
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache using configured strategy"""
        
        ttl = ttl or self.default_ttl
        
        if self.cache_strategy == CacheStrategy.WRITE_THROUGH:
            # Write to all levels
            await self._set_to_l1(key, value, ttl)
            await self._set_to_l2(key, value, ttl)
        elif self.cache_strategy == CacheStrategy.CACHE_ASIDE:
            # Write to L1 and L2
            await self._set_to_l1(key, value, ttl)
            await self._set_to_l2(key, value, ttl)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        
        # Delete from L1
        if key in self.l1_cache:
            del self.l1_cache[key]
            self.l1_stats.deletes += 1
        
        # Delete from L2
        await self._delete_from_l2(key)
        self.l2_stats.deletes += 1
        
        return True
    
    async def _get_from_l1(self, key: str) -> Any:
        """Get value from L1 cache"""
        
        if key not in self.l1_cache:
            return None
        
        entry = self.l1_cache[key]
        
        # Check TTL
        if self._is_expired(entry):
            del self.l1_cache[key]
            self.l1_stats.evictions += 1
            return None
        
        # Update access info
        entry.last_accessed = datetime.now(datetime.UTC)
        entry.access_count += 1
        
        return entry.value
    
    async def _set_to_l1(self, key: str, value: Any, ttl: int):
        """Set value in L1 cache"""
        
        # Check if we need to evict
        await self._evict_if_needed_l1()
        
        # Calculate size
        size_bytes = len(pickle.dumps(value))
        
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            created_at=datetime.now(datetime.UTC),
            last_accessed=datetime.now(datetime.UTC),
            size_bytes=size_bytes
        )
        
        self.l1_cache[key] = entry
        self.l1_stats.writes += 1
        self.l1_stats.memory_usage += size_bytes
    
    async def _get_from_l2(self, key: str) -> Any:
        """Get value from L2 cache (Redis)"""
        
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            cached_data = await redis.get(key)
            
            if cached_data:
                return pickle.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"L2 cache get error for key {key}: {e}")
            return None
    
    async def _set_to_l2(self, key: str, value: Any, ttl: int):
        """Set value in L2 cache (Redis)"""
        
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            serialized_value = pickle.dumps(value)
            await redis.setex(key, ttl, serialized_value)
            self.l2_stats.writes += 1
            
        except Exception as e:
            logger.error(f"L2 cache set error for key {key}: {e}")
    
    async def _delete_from_l2(self, key: str):
        """Delete value from L2 cache (Redis)"""
        
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            await redis.delete(key)
            
        except Exception as e:
            logger.error(f"L2 cache delete error for key {key}: {e}")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        
        age = (datetime.now(datetime.UTC) - entry.created_at).total_seconds()
        return age > entry.ttl
    
    async def _evict_if_needed_l1(self):
        """Evict entries from L1 cache if needed"""
        
        # Check size limits
        if len(self.l1_cache) >= self.l1_max_size or self.l1_stats.memory_usage >= self.l1_max_memory:
            await self._evict_l1_entries(max(1, len(self.l1_cache) // 10))
    
    async def _evict_l1_entries(self, count: int):
        """Evict entries from L1 cache based on policy"""
        
        if not self.l1_cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Evict least recently used
            sorted_entries = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].last_accessed
            )
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Evict least frequently used
            sorted_entries = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].access_count
            )
        else:  # FIFO
            # Evict oldest entries
            sorted_entries = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].created_at
            )
        
        # Evict entries
        for i in range(min(count, len(sorted_entries))):
            key, entry = sorted_entries[i]
            self.l1_stats.memory_usage -= entry.size_bytes
            del self.l1_cache[key]
            self.l1_stats.evictions += 1
    
    async def _cache_maintenance_loop(self):
        """Background task for cache maintenance"""
        
        while True:
            try:
                # Clean expired entries from L1
                expired_keys = []
                for key, entry in self.l1_cache.items():
                    if self._is_expired(entry):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    entry = self.l1_cache[key]
                    self.l1_stats.memory_usage -= entry.size_bytes
                    del self.l1_cache[key]
                    self.l1_stats.evictions += 1
                
                if expired_keys:
                    logger.debug(f"Cleaned {len(expired_keys)} expired entries from L1 cache")
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
                await asyncio.sleep(60)
    
    async def _cache_warming_loop(self):
        """Background task for cache warming"""
        
        while self.warming_enabled:
            try:
                # Execute warming tasks
                for task_name, task_func in self.warming_tasks.items():
                    try:
                        await task_func()
                        logger.debug(f"Executed cache warming task: {task_name}")
                    except Exception as e:
                        logger.error(f"Cache warming task {task_name} failed: {e}")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(300)
    
    def add_warming_task(self, name: str, task_func: Callable):
        """Add a cache warming task"""
        
        self.warming_tasks[name] = task_func
        logger.info(f"Added cache warming task: {name}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        return {
            'l1_cache': {
                'entries': len(self.l1_cache),
                'memory_usage': self.l1_stats.memory_usage,
                'hit_rate': self.l1_stats.hit_rate,
                'hits': self.l1_stats.hits,
                'misses': self.l1_stats.misses,
                'evictions': self.l1_stats.evictions
            },
            'l2_cache': {
                'hit_rate': self.l2_stats.hit_rate,
                'hits': self.l2_stats.hits,
                'misses': self.l2_stats.misses,
                'writes': self.l2_stats.writes
            },
            'configuration': {
                'eviction_policy': self.eviction_policy.value,
                'cache_strategy': self.cache_strategy.value,
                'default_ttl': self.default_ttl,
                'l1_max_size': self.l1_max_size,
                'l1_max_memory': self.l1_max_memory
            }
        }

class CacheInvalidationManager:
    """Manages cache invalidation strategies"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.invalidation_patterns = {}
        self.dependency_graph = defaultdict(set)
    
    def register_invalidation_pattern(self, pattern: str, keys: List[str]):
        """Register cache invalidation pattern"""
        
        self.invalidation_patterns[pattern] = keys
        logger.info(f"Registered invalidation pattern: {pattern}")
    
    def add_dependency(self, parent_key: str, dependent_key: str):
        """Add cache dependency relationship"""
        
        self.dependency_graph[parent_key].add(dependent_key)
    
    async def invalidate_pattern(self, pattern: str, **kwargs):
        """Invalidate cache entries matching pattern"""
        
        if pattern not in self.invalidation_patterns:
            logger.warning(f"Unknown invalidation pattern: {pattern}")
            return
        
        keys_to_invalidate = []
        
        for key_template in self.invalidation_patterns[pattern]:
            # Format key with provided kwargs
            try:
                key = key_template.format(**kwargs)
                keys_to_invalidate.append(key)
                
                # Add dependent keys
                if key in self.dependency_graph:
                    keys_to_invalidate.extend(self.dependency_graph[key])
                    
            except KeyError as e:
                logger.warning(f"Missing parameter for key template {key_template}: {e}")
        
        # Invalidate all keys
        for key in keys_to_invalidate:
            await self.cache.delete(key)
        
        logger.info(f"Invalidated {len(keys_to_invalidate)} cache entries for pattern {pattern}")

# Global cache instances
multi_level_cache = MultiLevelCache()
cache_invalidation_manager = CacheInvalidationManager(multi_level_cache)