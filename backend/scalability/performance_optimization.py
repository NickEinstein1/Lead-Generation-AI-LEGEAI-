"""
Performance Optimization Suite for Insurance Lead Scoring Platform

Provides comprehensive performance optimization including query optimization,
resource management, bottleneck detection, and automated performance tuning.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import psutil
import time
import threading
from collections import deque, defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

class BottleneckType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"

class OptimizationStrategy(Enum):
    SCALE_UP = "scale_up"
    SCALE_OUT = "scale_out"
    CACHE_OPTIMIZATION = "cache_optimization"
    QUERY_OPTIMIZATION = "query_optimization"
    RESOURCE_REALLOCATION = "resource_reallocation"

@dataclass
class PerformanceMetric:
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: float
    threshold_critical: float
    
    @property
    def status(self) -> str:
        if self.value >= self.threshold_critical:
            return "critical"
        elif self.value >= self.threshold_warning:
            return "warning"
        else:
            return "normal"

@dataclass
class BottleneckDetection:
    bottleneck_type: BottleneckType
    severity: float  # 0-1 scale
    description: str
    suggested_actions: List[str]
    detected_at: datetime = field(default_factory=datetime.utcnow)

class PerformanceOptimizer:
    """Advanced performance optimization and bottleneck detection"""
    
    def __init__(self):
        self.metrics_history = defaultdict(deque)
        self.bottlenecks = []
        self.optimization_rules = {}
        self.performance_baselines = {}
        
        # Resource monitoring
        self.cpu_threshold_warning = 70.0
        self.cpu_threshold_critical = 85.0
        self.memory_threshold_warning = 80.0
        self.memory_threshold_critical = 90.0
        
        # Executors for parallel processing
        self.thread_executor = ThreadPoolExecutor(max_workers=10)
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        
        # Performance optimization state
        self.optimization_enabled = True
        self.auto_scaling_enabled = True
        
    async def start_monitoring(self):
        """Start performance monitoring"""
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._system_monitoring_loop()),
            asyncio.create_task(self._bottleneck_detection_loop()),
            asyncio.create_task(self._optimization_loop()),
            asyncio.create_task(self._baseline_update_loop())
        ]
        
        logger.info("Performance monitoring started")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
    
    async def _system_monitoring_loop(self):
        """Monitor system performance metrics"""
        
        while True:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_metric = PerformanceMetric(
                    metric_name="cpu_usage",
                    value=cpu_percent,
                    unit="%",
                    timestamp=datetime.now(datetime.UTC),
                    threshold_warning=self.cpu_threshold_warning,
                    threshold_critical=self.cpu_threshold_critical
                )
                self._record_metric(cpu_metric)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                memory_metric = PerformanceMetric(
                    metric_name="memory_usage",
                    value=memory.percent,
                    unit="%",
                    timestamp=datetime.now(datetime.UTC),
                    threshold_warning=self.memory_threshold_warning,
                    threshold_critical=self.memory_threshold_critical
                )
                self._record_metric(memory_metric)
                
                # Disk I/O metrics
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_metric = PerformanceMetric(
                        metric_name="disk_io_utilization",
                        value=disk_io.read_bytes + disk_io.write_bytes,
                        unit="bytes",
                        timestamp=datetime.now(datetime.UTC),
                        threshold_warning=1000000000,  # 1GB
                        threshold_critical=5000000000  # 5GB
                    )
                    self._record_metric(disk_metric)
                
                # Network I/O metrics
                network_io = psutil.net_io_counters()
                if network_io:
                    network_metric = PerformanceMetric(
                        metric_name="network_io_utilization",
                        value=network_io.bytes_sent + network_io.bytes_recv,
                        unit="bytes",
                        timestamp=datetime.now(datetime.UTC),
                        threshold_warning=1000000000,  # 1GB
                        threshold_critical=5000000000  # 5GB
                    )
                    self._record_metric(network_metric)
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(10)
    
    def _record_metric(self, metric: PerformanceMetric):
        """Record performance metric"""
        
        # Keep last 1000 metrics per type
        self.metrics_history[metric.metric_name].append(metric)
        if len(self.metrics_history[metric.metric_name]) > 1000:
            self.metrics_history[metric.metric_name].popleft()
        
        # Log critical metrics
        if metric.status == "critical":
            logger.warning(f"Critical performance metric: {metric.metric_name} = {metric.value}{metric.unit}")
    
    async def _bottleneck_detection_loop(self):
        """Detect performance bottlenecks"""
        
        while True:
            try:
                current_bottlenecks = []
                
                # Analyze CPU bottlenecks
                cpu_bottleneck = await self._detect_cpu_bottleneck()
                if cpu_bottleneck:
                    current_bottlenecks.append(cpu_bottleneck)
                
                # Analyze memory bottlenecks
                memory_bottleneck = await self._detect_memory_bottleneck()
                if memory_bottleneck:
                    current_bottlenecks.append(memory_bottleneck)
                
                # Analyze I/O bottlenecks
                io_bottleneck = await self._detect_io_bottleneck()
                if io_bottleneck:
                    current_bottlenecks.append(io_bottleneck)
                
                # Update bottlenecks list
                self.bottlenecks = current_bottlenecks
                
                if current_bottlenecks:
                    logger.info(f"Detected {len(current_bottlenecks)} performance bottlenecks")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Bottleneck detection error: {e}")
                await asyncio.sleep(30)
    
    async def _detect_cpu_bottleneck(self) -> Optional[BottleneckDetection]:
        """Detect CPU bottlenecks"""
        
        if "cpu_usage" not in self.metrics_history:
            return None
        
        recent_metrics = list(self.metrics_history["cpu_usage"])[-10:]  # Last 10 readings
        if len(recent_metrics) < 5:
            return None
        
        avg_cpu = sum(m.value for m in recent_metrics) / len(recent_metrics)
        
        if avg_cpu > self.cpu_threshold_critical:
            return BottleneckDetection(
                bottleneck_type=BottleneckType.CPU,
                severity=min(1.0, avg_cpu / 100.0),
                description=f"High CPU usage detected: {avg_cpu:.1f}%",
                suggested_actions=[
                    "Scale out to additional instances",
                    "Optimize CPU-intensive operations",
                    "Implement caching for expensive computations",
                    "Consider upgrading CPU resources"
                ]
            )
        
        return None
    
    async def _detect_memory_bottleneck(self) -> Optional[BottleneckDetection]:
        """Detect memory bottlenecks"""
        
        if "memory_usage" not in self.metrics_history:
            return None
        
        recent_metrics = list(self.metrics_history["memory_usage"])[-10:]
        if len(recent_metrics) < 5:
            return None
        
        avg_memory = sum(m.value for m in recent_metrics) / len(recent_metrics)
        
        if avg_memory > self.memory_threshold_critical:
            return BottleneckDetection(
                bottleneck_type=BottleneckType.MEMORY,
                severity=min(1.0, avg_memory / 100.0),
                description=f"High memory usage detected: {avg_memory:.1f}%",
                suggested_actions=[
                    "Optimize memory usage in applications",
                    "Implement memory-efficient data structures",
                    "Add more RAM to instances",
                    "Implement data pagination and streaming"
                ]
            )
        
        return None
    
    async def _detect_io_bottleneck(self) -> Optional[BottleneckDetection]:
        """Detect I/O bottlenecks"""
        
        disk_metrics = list(self.metrics_history.get("disk_io_utilization", []))[-5:]
        network_metrics = list(self.metrics_history.get("network_io_utilization", []))[-5:]
        
        if len(disk_metrics) >= 2:
            # Calculate I/O rate
            recent_disk = disk_metrics[-1].value
            previous_disk = disk_metrics[-2].value
            time_diff = (disk_metrics[-1].timestamp - disk_metrics[-2].timestamp).total_seconds()
            
            if time_diff > 0:
                disk_rate = (recent_disk - previous_disk) / time_diff
                
                if disk_rate > 100000000:  # 100MB/s threshold
                    return BottleneckDetection(
                        bottleneck_type=BottleneckType.DISK_IO,
                        severity=min(1.0, disk_rate / 1000000000),  # Normalize to 1GB/s
                        description=f"High disk I/O detected: {disk_rate/1000000:.1f} MB/s",
                        suggested_actions=[
                            "Optimize database queries",
                            "Implement SSD storage",
                            "Add read replicas",
                            "Implement data compression"
                        ]
                    )
        
        return None
    
    async def _optimization_loop(self):
        """Automatic optimization loop"""
        
        while self.optimization_enabled:
            try:
                if self.bottlenecks:
                    for bottleneck in self.bottlenecks:
                        await self._apply_optimization(bottleneck)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)
    
    async def _apply_optimization(self, bottleneck: BottleneckDetection):
        """Apply optimization for detected bottleneck"""
        
        if bottleneck.bottleneck_type == BottleneckType.CPU:
            await self._optimize_cpu_usage()
        elif bottleneck.bottleneck_type == BottleneckType.MEMORY:
            await self._optimize_memory_usage()
        elif bottleneck.bottleneck_type == BottleneckType.DISK_IO:
            await self._optimize_disk_io()
        
        logger.info(f"Applied optimization for {bottleneck.bottleneck_type.value} bottleneck")
    
    async def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        
        # Implement CPU optimization strategies
        # This could include:
        # - Adjusting thread pool sizes
        # - Enabling CPU-intensive task queuing
        # - Triggering auto-scaling
        
        if self.auto_scaling_enabled:
            # Trigger scaling (would integrate with auto-scaling system)
            logger.info("Triggering CPU-based auto-scaling")
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage"""
        
        # Implement memory optimization strategies
        # This could include:
        # - Forcing garbage collection
        # - Clearing caches
        # - Reducing cache sizes
        
        logger.info("Applying memory optimization strategies")
    
    async def _optimize_disk_io(self):
        """Optimize disk I/O"""
        
        # Implement I/O optimization strategies
        # This could include:
        # - Enabling query result caching
        # - Optimizing database connections
        # - Implementing read replicas
        
        logger.info("Applying disk I/O optimization strategies")
    
    async def _baseline_update_loop(self):
        """Update performance baselines"""
        
        while True:
            try:
                # Update baselines every hour
                await self._update_performance_baselines()
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Baseline update error: {e}")
                await asyncio.sleep(3600)
    
    async def _update_performance_baselines(self):
        """Update performance baselines based on historical data"""
        
        for metric_name, metrics in self.metrics_history.items():
            if len(metrics) >= 100:  # Need sufficient data
                values = [m.value for m in metrics]
                
                self.performance_baselines[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'updated_at': datetime.now(datetime.UTC)
                }
        
        logger.debug("Updated performance baselines")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'timestamp': datetime.now(datetime.UTC).isoformat(),
            'current_metrics': {},
            'bottlenecks': [],
            'baselines': self.performance_baselines,
            'recommendations': []
        }
        
        # Current metrics
        for metric_name, metrics in self.metrics_history.items():
            if metrics:
                latest_metric = metrics[-1]
                report['current_metrics'][metric_name] = {
                    'value': latest_metric.value,
                    'unit': latest_metric.unit,
                    'status': latest_metric.status,
                    'timestamp': latest_metric.timestamp.isoformat()
                }
        
        # Bottlenecks
        for bottleneck in self.bottlenecks:
            report['bottlenecks'].append({
                'type': bottleneck.bottleneck_type.value,
                'severity': bottleneck.severity,
                'description': bottleneck.description,
                'suggested_actions': bottleneck.suggested_actions,
                'detected_at': bottleneck.detected_at.isoformat()
            })
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        
        recommendations = []
        
        # Check if any metrics are consistently high
        for metric_name, metrics in self.metrics_history.items():
            if len(metrics) >= 10:
                recent_values = [m.value for m in list(metrics)[-10:]]
                avg_value = sum(recent_values) / len(recent_values)
                
                if metric_name == "cpu_usage" and avg_value > 60:
                    recommendations.append("Consider CPU optimization or scaling")
                elif metric_name == "memory_usage" and avg_value > 70:
                    recommendations.append("Consider memory optimization or additional RAM")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System performance is within normal parameters")
        
        return recommendations

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()