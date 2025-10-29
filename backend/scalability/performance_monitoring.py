import asyncio
import psutil
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import redis

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Performance alert"""
    id: str
    metric_name: str
    threshold: float
    current_value: float
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 3
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Metrics storage
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: Dict[str, Alert] = {}
        
        # Monitoring configuration
        self.collection_interval = 10  # seconds
        self.alert_thresholds = self._load_alert_thresholds()
        
        # State
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance counters
        self.request_counter = 0
        self.error_counter = 0
        self.response_times = deque(maxlen=1000)
        
    def _load_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load alert thresholds configuration"""
        return {
            'cpu_usage': {
                'warning': 70.0,
                'critical': 90.0
            },
            'memory_usage': {
                'warning': 80.0,
                'critical': 95.0
            },
            'disk_usage': {
                'warning': 85.0,
                'critical': 95.0
            },
            'response_time': {
                'warning': 1.0,
                'critical': 3.0
            },
            'error_rate': {
                'warning': 0.05,  # 5%
                'critical': 0.10  # 10%
            },
            'queue_length': {
                'warning': 100,
                'critical': 500
            }
        }
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        self.running = True
        logger.info("Performance monitoring started")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._system_metrics_loop()),
            asyncio.create_task(self._application_metrics_loop()),
            asyncio.create_task(self._alert_processing_loop()),
            asyncio.create_task(self._metrics_persistence_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.running = False
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Performance monitoring stopped")
    
    async def _system_metrics_loop(self):
        """Collect system metrics"""
        while self.running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _application_metrics_loop(self):
        """Collect application metrics"""
        while self.running:
            try:
                await self._collect_application_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Application metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _alert_processing_loop(self):
        """Process alerts"""
        while self.running:
            try:
                await self._process_alerts()
                await asyncio.sleep(30)  # Check alerts every 30 seconds
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_persistence_loop(self):
        """Persist metrics to Redis"""
        while self.running:
            try:
                await self._persist_metrics()
                await asyncio.sleep(60)  # Persist every minute
            except Exception as e:
                logger.error(f"Metrics persistence error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        self._record_metric('cpu_usage', cpu_percent, timestamp)
        self._record_metric('cpu_count', cpu_count, timestamp)
        self._record_metric('load_avg_1m', load_avg[0], timestamp)
        self._record_metric('load_avg_5m', load_avg[1], timestamp)
        self._record_metric('load_avg_15m', load_avg[2], timestamp)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        self._record_metric('memory_usage', memory.percent, timestamp)
        self._record_metric('memory_total', memory.total, timestamp)
        self._record_metric('memory_available', memory.available, timestamp)
        self._record_metric('swap_usage', swap.percent, timestamp)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        self._record_metric('disk_usage', (disk.used / disk.total) * 100, timestamp)
        self._record_metric('disk_total', disk.total, timestamp)
        self._record_metric('disk_free', disk.free, timestamp)
        
        if disk_io:
            self._record_metric('disk_read_bytes', disk_io.read_bytes, timestamp)
            self._record_metric('disk_write_bytes', disk_io.write_bytes, timestamp)
        
        # Network metrics
        network_io = psutil.net_io_counters()
        if network_io:
            self._record_metric('network_bytes_sent', network_io.bytes_sent, timestamp)
            self._record_metric('network_bytes_recv', network_io.bytes_recv, timestamp)
            self._record_metric('network_packets_sent', network_io.packets_sent, timestamp)
            self._record_metric('network_packets_recv', network_io.packets_recv, timestamp)
        
        # Process metrics
        process = psutil.Process()
        self._record_metric('process_cpu_percent', process.cpu_percent(), timestamp)
        self._record_metric('process_memory_percent', process.memory_percent(), timestamp)
        self._record_metric('process_num_threads', process.num_threads(), timestamp)
        self._record_metric('process_num_fds', process.num_fds() if hasattr(process, 'num_fds') else 0, timestamp)
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        timestamp = datetime.now()
        
        try:
            # Queue metrics
            from backend.scalability.distributed_processing import task_queue, worker_manager
            
            queue_stats = task_queue.get_queue_stats()
            worker_stats = worker_manager.get_worker_stats()
            
            self._record_metric('queue_total_pending', queue_stats.get('total_pending', 0), timestamp)
            self._record_metric('queue_urgent', queue_stats.get('urgent', 0), timestamp)
            self._record_metric('queue_high', queue_stats.get('high', 0), timestamp)
            self._record_metric('queue_normal', queue_stats.get('normal', 0), timestamp)
            self._record_metric('queue_low', queue_stats.get('low', 0), timestamp)
            
            self._record_metric('workers_total', worker_stats.get('total_workers', 0), timestamp)
            self._record_metric('workers_tasks_processed', 
                              worker_stats['aggregate_stats'].get('total_tasks_processed', 0), timestamp)
            self._record_metric('workers_tasks_failed', 
                              worker_stats['aggregate_stats'].get('total_tasks_failed', 0), timestamp)
            self._record_metric('workers_avg_cpu', 
                              worker_stats['aggregate_stats'].get('avg_cpu_usage', 0), timestamp)
            self._record_metric('workers_avg_memory', 
                              worker_stats['aggregate_stats'].get('avg_memory_usage', 0), timestamp)
            
        except Exception as e:
            logger.error(f"Failed to collect queue metrics: {e}")
        
        try:
            # Cache metrics
            from backend.scalability.load_balancing import distributed_cache
            
            cache_stats = distributed_cache.get_stats()
            
            self._record_metric('cache_hits', cache_stats.get('cache_hits', 0), timestamp)
            self._record_metric('cache_misses', cache_stats.get('cache_misses', 0), timestamp)
            self._record_metric('cache_hit_rate', cache_stats.get('hit_rate', 0), timestamp)
            self._record_metric('cache_total_keys', cache_stats.get('total_keys', 0), timestamp)
            
        except Exception as e:
            logger.error(f"Failed to collect cache metrics: {e}")
        
        try:
            # Load balancer metrics
            from backend.scalability.load_balancing import api_load_balancer
            
            lb_stats = api_load_balancer.get_stats()
            
            self._record_metric('lb_total_servers', lb_stats.get('total_servers', 0), timestamp)
            self._record_metric('lb_healthy_servers', lb_stats.get('healthy_servers', 0), timestamp)
            self._record_metric('lb_total_requests', lb_stats.get('total_requests', 0), timestamp)
            self._record_metric('lb_error_rate', lb_stats.get('error_rate', 0), timestamp)
            self._record_metric('lb_avg_response_time', lb_stats.get('avg_response_time', 0), timestamp)
            
        except Exception as e:
            logger.error(f"Failed to collect load balancer metrics: {e}")
        
        # Application performance metrics
        if self.response_times:
            avg_response_time = statistics.mean(self.response_times)
            p95_response_time = statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(self.response_times, n=100)[98]  # 99th percentile
            
            self._record_metric('app_avg_response_time', avg_response_time, timestamp)
            self._record_metric('app_p95_response_time', p95_response_time, timestamp)
            self._record_metric('app_p99_response_time', p99_response_time, timestamp)
        
        # Error rate
        total_requests = self.request_counter
        if total_requests > 0:
            error_rate = self.error_counter / total_requests
            self._record_metric('app_error_rate', error_rate, timestamp)
        
        self._record_metric('app_total_requests', total_requests, timestamp)
        self._record_metric('app_total_errors', self.error_counter, timestamp)
    
    def _record_metric(self, metric_name: str, value: float, timestamp: datetime, 
                      tags: Dict[str, str] = None):
        """Record a metric value"""
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value,
            tags=tags or {}
        )
        
        self.metrics_buffer[metric_name].append(metric)
    
    async def _process_alerts(self):
        """Process alerts based on current metrics"""
        current_time = datetime.now()
        
        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name not in self.metrics_buffer:
                continue
            
            # Get recent metrics (last 5 minutes)
            cutoff_time = current_time - timedelta(minutes=5)
            recent_metrics = [
                m for m in self.metrics_buffer[metric_name]
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                continue
            
            # Calculate average value
            avg_value = statistics.mean([m.value for m in recent_metrics])
            
            # Check thresholds
            alert_id = f"{metric_name}_alert"
            
            if avg_value >= thresholds.get('critical', float('inf')):
                await self._create_alert(
                    alert_id, metric_name, avg_value, 
                    thresholds['critical'], 'critical'
                )
            elif avg_value >= thresholds.get('warning', float('inf')):
                await self._create_alert(
                    alert_id, metric_name, avg_value, 
                    thresholds['warning'], 'warning'
                )
            else:
                # Resolve alert if it exists
                await self._resolve_alert(alert_id)
    
    async def _create_alert(self, alert_id: str, metric_name: str, 
                          current_value: float, threshold: float, severity: str):
        """Create or update an alert"""
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            # Update existing alert
            self.alerts[alert_id].current_value = current_value
            self.alerts[alert_id].timestamp = datetime.now()
        else:
            # Create new alert
            alert = Alert(
                id=alert_id,
                metric_name=metric_name,
                threshold=threshold,
                current_value=current_value,
                severity=severity,
                message=f"{metric_name} is {current_value:.2f}, exceeding {severity} threshold of {threshold}",
                timestamp=datetime.now()
            )
            
            self.alerts[alert_id] = alert
            
            # Log alert
            logger.warning(f"ALERT [{severity.upper()}]: {alert.message}")
            
            # Send notification (implement as needed)
            await self._send_alert_notification(alert)
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.now()
            
            logger.info(f"RESOLVED: Alert {alert_id}")
    
    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification (implement based on requirements)"""
        # This could send emails, Slack messages, webhooks, etc.
        # For now, just log
        logger.critical(f"ALERT NOTIFICATION: {alert.message}")
    
    async def _persist_metrics(self):
        """Persist metrics to Redis for long-term storage"""
        try:
            current_time = datetime.now()
            
            for metric_name, metrics in self.metrics_buffer.items():
                # Get metrics from last minute
                cutoff_time = current_time - timedelta(minutes=1)
                recent_metrics = [
                    m for m in metrics
                    if m.timestamp >= cutoff_time
                ]
                
                if not recent_metrics:
                    continue
                
                # Calculate aggregated values
                values = [m.value for m in recent_metrics]
                aggregated_data = {
                    'timestamp': current_time.isoformat(),
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values),
                    'sum': sum(values)
                }
                
                if len(values) > 1:
                    aggregated_data['std'] = statistics.stdev(values)
                
                # Store in Redis with TTL (keep for 7 days)
                key = f"metrics:{metric_name}:{int(current_time.timestamp())}"
                self.redis_client.setex(
                    key, 
                    7 * 24 * 3600,  # 7 days
                    json.dumps(aggregated_data)
                )
            
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request for performance tracking"""
        self.request_counter += 1
        self.response_times.append(response_time)
        
        if not success:
            self.error_counter += 1
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=5)
        
        metrics_summary = {}
        
        for metric_name, metrics in self.metrics_buffer.items():
            recent_metrics = [
                m for m in metrics
                if m.timestamp >= cutoff_time
            ]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                metrics_summary[metric_name] = {
                    'current': values[-1],
                    'avg_5m': statistics.mean(values),
                    'min_5m': min(values),
                    'max_5m': max(values),
                    'count': len(values)
                }
        
        return metrics_summary
    
    def get_alerts(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Get current alerts"""
        alerts = []
        
        for alert in self.alerts.values():
            if not include_resolved and alert.resolved:
                continue
            
            alerts.append({
                'id': alert.id,
                'metric_name': alert.metric_name,
                'threshold': alert.threshold,
                'current_value': alert.current_value,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'resolved': alert.resolved,
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
            })
        
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current_metrics = self.get_current_metrics()
        active_alerts = self.get_alerts(include_resolved=False)
        
        # Calculate health score (0-100)
        health_score = 100
        
        # Deduct points for alerts
        for alert in active_alerts:
            if alert['severity'] == 'critical':
                health_score -= 20
            elif alert['severity'] == 'warning':
                health_score -= 10
        
        # Deduct points for high resource usage
        if 'cpu_usage' in current_metrics:
            cpu_usage = current_metrics['cpu_usage']['current']
            if cpu_usage > 80:
                health_score -= (cpu_usage - 80) / 2
        
        if 'memory_usage' in current_metrics:
            memory_usage = current_metrics['memory_usage']['current']
            if memory_usage > 80:
                health_score -= (memory_usage - 80) / 2
        
        health_score = max(0, health_score)
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 50 else 'unhealthy',
            'active_alerts': len(active_alerts),
            'total_requests': self.request_counter,
            'total_errors': self.error_counter,
            'error_rate': self.error_counter / self.request_counter if self.request_counter > 0 else 0,
            'avg_response_time': statistics.mean(self.response_times) if self.response_times else 0,
            'uptime': (datetime.now() - datetime.now()).total_seconds(),  # Would track actual start time
            'current_metrics': current_metrics,
            'alerts': active_alerts
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Decorator for monitoring function performance
def monitor_performance(func_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func: Callable):
        name = func_name or f"{func.__module__}.{func.__name__}"
        
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                response_time = time.time() - start_time
                performance_monitor.record_request(response_time, success)
                
                # Record function-specific metrics
                timestamp = datetime.now()
                performance_monitor._record_metric(f'func_{name}_response_time', response_time, timestamp)
                performance_monitor._record_metric(f'func_{name}_success', 1 if success else 0, timestamp)
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                response_time = time.time() - start_time
                performance_monitor.record_request(response_time, success)
                
                # Record function-specific metrics
                timestamp = datetime.now()
                performance_monitor._record_metric(f'func_{name}_response_time', response_time, timestamp)
                performance_monitor._record_metric(f'func_{name}_success', 1 if success else 0, timestamp)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# Example usage
async def example_usage():
    """Example of using performance monitoring"""
    
    # Start monitoring
    monitor_task = asyncio.create_task(performance_monitor.start_monitoring())
    
    try:
        # Simulate some load
        for i in range(100):
            # Simulate request
            response_time = random.uniform(0.1, 2.0)
            success = random.random() > 0.05  # 5% error rate
            
            performance_monitor.record_request(response_time, success)
            
            await asyncio.sleep(0.1)
        
        # Wait a bit for metrics collection
        await asyncio.sleep(30)
        
        # Get performance summary
        summary = performance_monitor.get_performance_summary()
        print("Performance Summary:")
        print(json.dumps(summary, indent=2, default=str))
        
    finally:
        await performance_monitor.stop_monitoring()
        monitor_task.cancel()

if __name__ == "__main__":
    import random
    asyncio.run(example_usage())
