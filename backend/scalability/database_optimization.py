"""
Database Optimization and Scaling for Insurance Lead Scoring Platform

Provides database connection pooling, query optimization, read replicas,
sharding strategies, and database performance monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncpg
import aioredis
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
import psutil
import time

logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    REDIS = "redis"
    MONGODB = "mongodb"

class QueryType(Enum):
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    BULK_INSERT = "bulk_insert"

@dataclass
class DatabaseConfig:
    db_type: DatabaseType
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    read_replicas: List[str] = field(default_factory=list)

@dataclass
class QueryMetrics:
    query_id: str
    query_type: QueryType
    execution_time: float
    rows_affected: int
    database: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

class DatabaseConnectionManager:
    """Advanced database connection management with pooling and optimization"""
    
    def __init__(self):
        self.connection_pools = {}
        self.read_replica_pools = {}
        self.query_metrics = []
        self.slow_query_threshold = 1.0  # seconds
        self.connection_stats = {}
        
    async def initialize_database_pools(self, configs: List[DatabaseConfig]):
        """Initialize database connection pools"""
        
        for config in configs:
            try:
                if config.db_type == DatabaseType.POSTGRESQL:
                    await self._setup_postgresql_pool(config)
                elif config.db_type == DatabaseType.REDIS:
                    await self._setup_redis_pool(config)
                
                logger.info(f"Initialized {config.db_type.value} pool for {config.database}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {config.db_type.value} pool: {e}")
                raise
    
    async def _setup_postgresql_pool(self, config: DatabaseConfig):
        """Setup PostgreSQL connection pool"""
        
        # Main database pool
        dsn = f"postgresql://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"
        
        pool = await asyncpg.create_pool(
            dsn,
            min_size=config.pool_size // 2,
            max_size=config.pool_size,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=60
        )
        
        self.connection_pools[config.database] = pool
        
        # Read replica pools
        for replica_host in config.read_replicas:
            replica_dsn = f"postgresql://{config.username}:{config.password}@{replica_host}:{config.port}/{config.database}"
            replica_pool = await asyncpg.create_pool(
                replica_dsn,
                min_size=config.pool_size // 4,
                max_size=config.pool_size // 2,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            
            if config.database not in self.read_replica_pools:
                self.read_replica_pools[config.database] = []
            self.read_replica_pools[config.database].append(replica_pool)
    
    async def _setup_redis_pool(self, config: DatabaseConfig):
        """Setup Redis connection pool"""
        
        redis_url = f"redis://{config.host}:{config.port}/{config.database}"
        pool = aioredis.ConnectionPool.from_url(
            redis_url,
            max_connections=config.pool_size,
            retry_on_timeout=True
        )
        
        self.connection_pools[f"redis_{config.database}"] = pool
    
    async def execute_query(self, database: str, query: str, params: Optional[List] = None, 
                          use_replica: bool = False) -> Any:
        """Execute database query with performance monitoring"""
        
        start_time = time.time()
        query_id = f"query_{int(start_time * 1000)}"
        
        try:
            # Choose connection pool
            if use_replica and database in self.read_replica_pools:
                # Use read replica for SELECT queries
                pool = self._get_best_replica_pool(database)
            else:
                pool = self.connection_pools[database]
            
            # Execute query
            async with pool.acquire() as connection:
                if params:
                    result = await connection.fetch(query, *params)
                else:
                    result = await connection.fetch(query)
            
            execution_time = time.time() - start_time
            
            # Record metrics
            metrics = QueryMetrics(
                query_id=query_id,
                query_type=self._determine_query_type(query),
                execution_time=execution_time,
                rows_affected=len(result) if result else 0,
                database=database
            )
            
            self.query_metrics.append(metrics)
            
            # Log slow queries
            if execution_time > self.slow_query_threshold:
                logger.warning(f"Slow query detected: {query_id} took {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed: {query_id} after {execution_time:.2f}s - {e}")
            raise
    
    def _get_best_replica_pool(self, database: str):
        """Get the best performing read replica pool"""
        
        if database not in self.read_replica_pools:
            return self.connection_pools[database]
        
        # Simple round-robin for now
        # Could be enhanced with health checking and load metrics
        replicas = self.read_replica_pools[database]
        if not hasattr(self, '_replica_index'):
            self._replica_index = {}
        
        if database not in self._replica_index:
            self._replica_index[database] = 0
        
        pool = replicas[self._replica_index[database]]
        self._replica_index[database] = (self._replica_index[database] + 1) % len(replicas)
        
        return pool
    
    def _determine_query_type(self, query: str) -> QueryType:
        """Determine query type from SQL"""
        
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        else:
            return QueryType.SELECT
    
    async def bulk_insert(self, database: str, table: str, data: List[Dict[str, Any]]) -> int:
        """Optimized bulk insert operation"""
        
        if not data:
            return 0
        
        start_time = time.time()
        
        try:
            pool = self.connection_pools[database]
            
            async with pool.acquire() as connection:
                # Prepare bulk insert
                columns = list(data[0].keys())
                placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
                query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                
                # Execute bulk insert
                values_list = [[row[col] for col in columns] for row in data]
                await connection.executemany(query, values_list)
            
            execution_time = time.time() - start_time
            
            # Record metrics
            metrics = QueryMetrics(
                query_id=f"bulk_insert_{int(start_time * 1000)}",
                query_type=QueryType.BULK_INSERT,
                execution_time=execution_time,
                rows_affected=len(data),
                database=database
            )
            
            self.query_metrics.append(metrics)
            
            logger.info(f"Bulk inserted {len(data)} rows in {execution_time:.2f}s")
            return len(data)
            
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics"""
        
        if not self.query_metrics:
            return {}
        
        # Calculate statistics
        recent_metrics = [m for m in self.query_metrics if m.timestamp > datetime.now(datetime.UTC) - timedelta(hours=1)]
        
        stats = {
            'total_queries': len(self.query_metrics),
            'recent_queries': len(recent_metrics),
            'avg_execution_time': sum(m.execution_time for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
            'slow_queries': len([m for m in recent_metrics if m.execution_time > self.slow_query_threshold]),
            'queries_by_type': {},
            'connection_pools': {}
        }
        
        # Query type breakdown
        for query_type in QueryType:
            count = len([m for m in recent_metrics if m.query_type == query_type])
            stats['queries_by_type'][query_type.value] = count
        
        # Connection pool stats
        for db_name, pool in self.connection_pools.items():
            if hasattr(pool, 'get_size'):
                stats['connection_pools'][db_name] = {
                    'size': pool.get_size(),
                    'free_size': pool.get_idle_size() if hasattr(pool, 'get_idle_size') else 0
                }
        
        return stats

class DatabaseShardingManager:
    """Database sharding management for horizontal scaling"""
    
    def __init__(self, connection_manager: DatabaseConnectionManager):
        self.connection_manager = connection_manager
        self.shard_configs = {}
        self.shard_mappings = {}
    
    def configure_sharding(self, table: str, shard_key: str, shard_configs: List[Dict[str, Any]]):
        """Configure sharding for a table"""
        
        self.shard_configs[table] = {
            'shard_key': shard_key,
            'shards': shard_configs
        }
        
        logger.info(f"Configured sharding for {table} with {len(shard_configs)} shards")
    
    def get_shard_for_key(self, table: str, key_value: Any) -> str:
        """Determine which shard to use for a given key"""
        
        if table not in self.shard_configs:
            raise ValueError(f"No sharding configuration for table {table}")
        
        shards = self.shard_configs[table]['shards']
        
        # Simple hash-based sharding
        shard_index = hash(str(key_value)) % len(shards)
        return shards[shard_index]['database']
    
    async def execute_sharded_query(self, table: str, key_value: Any, query: str, 
                                  params: Optional[List] = None) -> Any:
        """Execute query on the appropriate shard"""
        
        shard_database = self.get_shard_for_key(table, key_value)
        return await self.connection_manager.execute_query(shard_database, query, params)
    
    async def execute_cross_shard_query(self, table: str, query: str, 
                                      params: Optional[List] = None) -> List[Any]:
        """Execute query across all shards and aggregate results"""
        
        if table not in self.shard_configs:
            raise ValueError(f"No sharding configuration for table {table}")
        
        shards = self.shard_configs[table]['shards']
        results = []
        
        # Execute on all shards
        tasks = []
        for shard in shards:
            task = self.connection_manager.execute_query(shard['database'], query, params)
            tasks.append(task)
        
        shard_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in shard_results:
            if isinstance(result, Exception):
                logger.error(f"Shard query failed: {result}")
            else:
                results.extend(result)
        
        return results

# Global database manager instance
db_manager = DatabaseConnectionManager()
sharding_manager = DatabaseShardingManager(db_manager)