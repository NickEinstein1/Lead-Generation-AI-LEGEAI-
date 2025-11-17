"""
Comprehensive Scalability and Performance Module

This module provides enterprise-grade scalability and performance optimization
for the Insurance Lead Scoring Platform, including:

- Distributed processing with auto-scaling workers
- Multi-level caching strategies
- Database optimization and sharding
- Performance monitoring and bottleneck detection
- Load balancing and circuit breakers
- Automated performance tuning
"""

from .distributed_processing import (
    task_queue, worker_manager, TaskQueue, DistributedWorker, WorkerManager,
    submit_lead_scoring_task, submit_batch_scoring_task, submit_enrichment_task
)
from .auto_scaling import (
    auto_scaler, metrics_collector, orchestrator, AutoScaler, MetricsCollector,
    start_auto_scaling
)
from .performance_monitoring import (
    performance_monitor, PerformanceMonitor
)
from .load_balancing import (
    api_load_balancer, distributed_cache, cached_lead_scorer,
    LoadBalancer, DistributedCache, CachedLeadScorer
)
from .database_optimization import (
    db_manager, sharding_manager, DatabaseConnectionManager, DatabaseShardingManager
)
from .caching_strategy import (
    multi_level_cache, cache_invalidation_manager, MultiLevelCache, CacheInvalidationManager
)
from .performance_optimization import (
    performance_optimizer, PerformanceOptimizer
)

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ScalabilityOrchestrator:
    """Central orchestrator for all scalability and performance components"""
    
    def __init__(self):
        self.components = {
            'task_queue': task_queue,
            'worker_manager': worker_manager,
            'auto_scaler': auto_scaler,
            'performance_monitor': performance_monitor,
            'load_balancer': api_load_balancer,
            'distributed_cache': distributed_cache,
            'db_manager': db_manager,
            'multi_level_cache': multi_level_cache,
            'performance_optimizer': performance_optimizer
        }
        
        self.initialized = False
        self.running = False
    
    async def initialize_all_components(self):
        """Initialize all scalability components"""
        
        try:
            logger.info("Initializing scalability and performance components...")
            
            # Initialize database connections
            await db_manager.initialize_database_pools([
                # Add your database configurations here
            ])
            
            # Initialize caching systems
            await multi_level_cache.initialize()
            
            # Initialize load balancer
            await api_load_balancer.initialize()
            
            # Set up cache warming tasks
            multi_level_cache.add_warming_task(
                "lead_scoring_cache_warm",
                self._warm_lead_scoring_cache
            )
            
            # Configure cache invalidation patterns
            cache_invalidation_manager.register_invalidation_pattern(
                "lead_update",
                ["lead_score:{lead_id}", "lead_data:{lead_id}", "batch_results:*"]
            )
            
            self.initialized = True
            logger.info("All scalability components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize scalability components: {e}")
            raise
    
    async def start_all_services(self):
        """Start all scalability services"""
        
        if not self.initialized:
            await self.initialize_all_components()
        
        try:
            logger.info("Starting scalability and performance services...")
            
            # Start all background services
            tasks = [
                asyncio.create_task(worker_manager.start()),
                asyncio.create_task(performance_monitor.start_monitoring()),
                asyncio.create_task(performance_optimizer.start_monitoring()),
                asyncio.create_task(self._start_auto_scaling()),
                asyncio.create_task(self._health_check_loop())
            ]
            
            self.running = True
            logger.info("All scalability services started successfully")
            
            # Wait for all services
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error starting scalability services: {e}")
            raise
    
    async def _start_auto_scaling(self):
        """Start auto-scaling services"""
        
        metrics_task, scaler_task = await start_auto_scaling()
        await asyncio.gather(metrics_task, scaler_task)
    
    async def _warm_lead_scoring_cache(self):
        """Cache warming task for lead scoring"""
        
        try:
            # This would typically pre-load frequently accessed lead scores
            # For now, just log the warming activity
            logger.debug("Executing lead scoring cache warming")
            
        except Exception as e:
            logger.error(f"Cache warming error: {e}")
    
    async def _health_check_loop(self):
        """Periodic health check for all components"""
        
        while self.running:
            try:
                health_status = await self.get_system_health()
                
                # Log any unhealthy components
                unhealthy_components = [
                    name for name, status in health_status['components'].items()
                    if status.get('status') != 'healthy'
                ]
                
                if unhealthy_components:
                    logger.warning(f"Unhealthy components detected: {unhealthy_components}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.now(datetime.UTC).isoformat(),
            'components': {}
        }
        
        try:
            # Check task queue health
            queue_stats = task_queue.get_queue_stats()
            health_status['components']['task_queue'] = {
                'status': 'healthy' if queue_stats.get('total_pending', 0) < 1000 else 'degraded',
                'pending_tasks': queue_stats.get('total_pending', 0),
                'workers': len(worker_manager.workers)
            }
            
            # Check cache health
            cache_stats = multi_level_cache.get_cache_stats()
            health_status['components']['cache'] = {
                'status': 'healthy' if cache_stats['l1_cache']['hit_rate'] > 0.5 else 'degraded',
                'l1_hit_rate': cache_stats['l1_cache']['hit_rate'],
                'l2_hit_rate': cache_stats['l2_cache']['hit_rate']
            }
            
            # Check database health
            db_stats = db_manager.get_performance_stats()
            health_status['components']['database'] = {
                'status': 'healthy' if db_stats.get('slow_queries', 0) < 10 else 'degraded',
                'recent_queries': db_stats.get('recent_queries', 0),
                'slow_queries': db_stats.get('slow_queries', 0)
            }
            
            # Check load balancer health
            lb_stats = api_load_balancer.get_stats()
            healthy_servers = len([s for s in lb_stats.get('servers', []) if s.get('status') == 'healthy'])
            total_servers = len(lb_stats.get('servers', []))
            
            health_status['components']['load_balancer'] = {
                'status': 'healthy' if healthy_servers > 0 else 'critical',
                'healthy_servers': healthy_servers,
                'total_servers': total_servers
            }
            
            # Check performance optimizer
            perf_report = performance_optimizer.get_performance_report()
            critical_bottlenecks = len([b for b in perf_report['bottlenecks'] if b['severity'] > 0.8])
            
            health_status['components']['performance'] = {
                'status': 'healthy' if critical_bottlenecks == 0 else 'warning',
                'bottlenecks': len(perf_report['bottlenecks']),
                'critical_bottlenecks': critical_bottlenecks
            }
            
            # Determine overall status
            component_statuses = [comp['status'] for comp in health_status['components'].values()]
            if 'critical' in component_statuses:
                health_status['overall_status'] = 'critical'
            elif 'degraded' in component_statuses or 'warning' in component_statuses:
                health_status['overall_status'] = 'degraded'
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
        
        return health_status
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        
        dashboard = {
            'timestamp': datetime.now(datetime.UTC).isoformat(),
            'system_health': await self.get_system_health(),
            'performance_metrics': performance_optimizer.get_performance_report(),
            'cache_statistics': multi_level_cache.get_cache_stats(),
            'database_performance': db_manager.get_performance_stats(),
            'load_balancer_stats': api_load_balancer.get_stats(),
            'worker_statistics': worker_manager.get_worker_stats(),
            'queue_statistics': task_queue.get_queue_stats()
        }
        
        return dashboard
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Trigger performance optimization across all components"""
        
        optimization_results = {
            'timestamp': datetime.now(datetime.UTC).isoformat(),
            'actions_taken': []
        }
        
        try:
            # Get current performance state
            perf_report = performance_optimizer.get_performance_report()
            
            # Apply optimizations based on bottlenecks
            for bottleneck in perf_report['bottlenecks']:
                if bottleneck['severity'] > 0.7:
                    # High severity bottleneck - take action
                    if bottleneck['type'] == 'cpu':
                        # Scale out workers
                        await worker_manager._start_worker()
                        optimization_results['actions_taken'].append('Added worker for CPU bottleneck')
                    
                    elif bottleneck['type'] == 'memory':
                        # Clear caches
                        await multi_level_cache.delete('*')
                        optimization_results['actions_taken'].append('Cleared caches for memory bottleneck')
            
            # Optimize cache hit rates
            cache_stats = multi_level_cache.get_cache_stats()
            if cache_stats['l1_cache']['hit_rate'] < 0.5:
                # Trigger cache warming
                await multi_level_cache._cache_warming_loop()
                optimization_results['actions_taken'].append('Triggered cache warming')
            
            logger.info(f"Performance optimization completed: {len(optimization_results['actions_taken'])} actions taken")
            
        except Exception as e:
            logger.error(f"Performance optimization error: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results

# Global scalability orchestrator
scalability_orchestrator = ScalabilityOrchestrator()

# Convenience functions
async def initialize_scalability_system():
    """Initialize the complete scalability system"""
    await scalability_orchestrator.initialize_all_components()

async def start_scalability_services():
    """Start all scalability services"""
    await scalability_orchestrator.start_all_services()

async def get_scalability_health():
    """Get scalability system health"""
    return await scalability_orchestrator.get_system_health()

async def get_performance_dashboard():
    """Get performance dashboard"""
    return await scalability_orchestrator.get_performance_dashboard()

# Export all components
__all__ = [
    # Core components
    'task_queue',
    'worker_manager', 
    'auto_scaler',
    'performance_monitor',
    'api_load_balancer',
    'distributed_cache',
    'db_manager',
    'multi_level_cache',
    'performance_optimizer',
    
    # Orchestrator
    'scalability_orchestrator',
    
    # Convenience functions
    'initialize_scalability_system',
    'start_scalability_services',
    'get_scalability_health',
    'get_performance_dashboard',
    
    # Classes
    'TaskQueue',
    'DistributedWorker',
    'WorkerManager',
    'AutoScaler',
    'MetricsCollector',
    'PerformanceMonitor',
    'LoadBalancer',
    'DistributedCache',
    'DatabaseConnectionManager',
    'MultiLevelCache',
    'PerformanceOptimizer',
    'ScalabilityOrchestrator',
    
    # Utility functions
    'submit_lead_scoring_task',
    'submit_batch_scoring_task',
    'submit_enrichment_task'
]

# Version info
__version__ = "2.0.0"
__author__ = "Insurance Lead Scoring Platform Scalability Team"