import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import time
from collections import defaultdict, deque
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class MetricType(Enum):
    """Types of metrics to monitor"""
    LATENCY = "LATENCY"
    THROUGHPUT = "THROUGHPUT"
    ERROR_RATE = "ERROR_RATE"
    ACCURACY = "ACCURACY"
    MEMORY_USAGE = "MEMORY_USAGE"
    CPU_USAGE = "CPU_USAGE"
    MODEL_DRIFT = "MODEL_DRIFT"
    DATA_QUALITY = "DATA_QUALITY"
    CONVERSION_RATE = "CONVERSION_RATE"
    REVENUE_IMPACT = "REVENUE_IMPACT"

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """Performance alert"""
    alert_id: str
    severity: AlertSeverity
    component: str
    metric_name: str
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class HealthCheck:
    """Component health check result"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    checks: Dict[str, bool]
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.metrics_buffer = deque(maxlen=10000)
        self.alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.health_status = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Metric aggregations
        self.metric_aggregations = defaultdict(lambda: defaultdict(list))
        self.rolling_windows = {
            '1m': deque(maxlen=60),
            '5m': deque(maxlen=300),
            '15m': deque(maxlen=900),
            '1h': deque(maxlen=3600)
        }
        
        # Thresholds
        self.thresholds = self.config.get('thresholds', {})
        self.baseline_metrics = {}
        
        # Model performance tracking
        self.model_performance = defaultdict(lambda: defaultdict(list))
        self.drift_detectors = {}
        
        logger.info("Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        # Add to buffer
        self.metrics_buffer.append(metric)
        
        # Add to rolling windows
        for window_name, window in self.rolling_windows.items():
            window.append(metric)
        
        # Add to aggregations
        key = f"{metric.component}.{metric.metric_name}"
        self.metric_aggregations[key][metric.metric_type].append(metric.value)
        
        # Check thresholds
        self._check_thresholds(metric)
        
        # Update model performance if applicable
        if metric.component.startswith('model_'):
            self._update_model_performance(metric)
    
    def record_api_call(self, endpoint: str, method: str, status_code: int, 
                       response_time: float, request_size: int = 0, 
                       response_size: int = 0):
        """Record API call metrics"""
        timestamp = datetime.now()
        
        # Latency metric
        latency_metric = PerformanceMetric(
            metric_name="response_time",
            metric_type=MetricType.LATENCY,
            value=response_time,
            timestamp=timestamp,
            component=f"api.{endpoint}",
            metadata={
                "method": method,
                "status_code": status_code,
                "request_size": request_size,
                "response_size": response_size
            },
            tags={"endpoint": endpoint, "method": method}
        )
        self.record_metric(latency_metric)
        
        # Error rate metric
        is_error = status_code >= 400
        error_metric = PerformanceMetric(
            metric_name="error_rate",
            metric_type=MetricType.ERROR_RATE,
            value=1.0 if is_error else 0.0,
            timestamp=timestamp,
            component=f"api.{endpoint}",
            metadata={"status_code": status_code},
            tags={"endpoint": endpoint}
        )
        self.record_metric(error_metric)
        
        # Throughput (requests per second)
        throughput_metric = PerformanceMetric(
            metric_name="requests_per_second",
            metric_type=MetricType.THROUGHPUT,
            value=1.0,  # Will be aggregated
            timestamp=timestamp,
            component=f"api.{endpoint}",
            tags={"endpoint": endpoint}
        )
        self.record_metric(throughput_metric)
    
    def record_model_prediction(self, model_name: str, prediction_time: float, 
                               input_features: Dict[str, Any], 
                               prediction: Any, confidence: float = None):
        """Record model prediction metrics"""
        timestamp = datetime.now()
        
        # Prediction latency
        latency_metric = PerformanceMetric(
            metric_name="prediction_time",
            metric_type=MetricType.LATENCY,
            value=prediction_time,
            timestamp=timestamp,
            component=f"model.{model_name}",
            metadata={
                "input_features": len(input_features),
                "prediction": str(prediction)[:100],  # Truncate for storage
                "confidence": confidence
            }
        )
        self.record_metric(latency_metric)
        
        # Model throughput
        throughput_metric = PerformanceMetric(
            metric_name="predictions_per_second",
            metric_type=MetricType.THROUGHPUT,
            value=1.0,
            timestamp=timestamp,
            component=f"model.{model_name}"
        )
        self.record_metric(throughput_metric)
        
        # Confidence tracking
        if confidence is not None:
            confidence_metric = PerformanceMetric(
                metric_name="prediction_confidence",
                metric_type=MetricType.ACCURACY,
                value=confidence,
                timestamp=timestamp,
                component=f"model.{model_name}"
            )
            self.record_metric(confidence_metric)
    
    def record_model_accuracy(self, model_name: str, actual_values: List[float], 
                             predicted_values: List[float], 
                             metric_name: str = "accuracy"):
        """Record model accuracy metrics"""
        if len(actual_values) != len(predicted_values):
            logger.error("Actual and predicted values must have same length")
            return
        
        # Calculate various accuracy metrics
        actual = np.array(actual_values)
        predicted = np.array(predicted_values)
        
        # R² Score
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        timestamp = datetime.now()
        
        # Record metrics
        for metric_value, metric_suffix in [(r2_score, "r2"), (mae, "mae"), (rmse, "rmse")]:
            accuracy_metric = PerformanceMetric(
                metric_name=f"{metric_name}_{metric_suffix}",
                metric_type=MetricType.ACCURACY,
                value=metric_value,
                timestamp=timestamp,
                component=f"model.{model_name}",
                metadata={
                    "sample_size": len(actual_values),
                    "actual_mean": np.mean(actual),
                    "predicted_mean": np.mean(predicted)
                }
            )
            self.record_metric(accuracy_metric)
    
    def record_system_metrics(self):
        """Record system resource metrics"""
        timestamp = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_metric = PerformanceMetric(
            metric_name="cpu_usage",
            metric_type=MetricType.CPU_USAGE,
            value=cpu_percent,
            timestamp=timestamp,
            component="system"
        )
        self.record_metric(cpu_metric)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_metric = PerformanceMetric(
            metric_name="memory_usage",
            metric_type=MetricType.MEMORY_USAGE,
            value=memory.percent,
            timestamp=timestamp,
            component="system",
            metadata={
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3)
            }
        )
        self.record_metric(memory_metric)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_metric = PerformanceMetric(
            metric_name="disk_usage",
            metric_type=MetricType.MEMORY_USAGE,
            value=(disk.used / disk.total) * 100,
            timestamp=timestamp,
            component="system",
            metadata={
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3)
            }
        )
        self.record_metric(disk_metric)
    
    def check_component_health(self, component: str) -> HealthCheck:
        """Perform health check on a component"""
        start_time = time.time()
        checks = {}
        status = "healthy"
        details = {}
        
        try:
            if component.startswith("api."):
                # API health checks
                checks["responsive"] = True  # Would make actual HTTP call
                checks["database_connection"] = True  # Would check DB
                checks["model_loaded"] = True  # Would check model status
                
            elif component.startswith("model."):
                # Model health checks
                model_name = component.split(".", 1)[1]
                checks["model_loaded"] = True  # Would check if model is loaded
                checks["prediction_latency"] = self._check_prediction_latency(model_name)
                checks["accuracy_threshold"] = self._check_accuracy_threshold(model_name)
                
            elif component == "system":
                # System health checks
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                
                checks["cpu_healthy"] = cpu_usage < 80
                checks["memory_healthy"] = memory_usage < 85
                checks["disk_healthy"] = psutil.disk_usage('/').percent < 90
                
                details["cpu_usage"] = cpu_usage
                details["memory_usage"] = memory_usage
                details["disk_usage"] = psutil.disk_usage('/').percent
            
            # Determine overall status
            if not all(checks.values()):
                failed_checks = [k for k, v in checks.items() if not v]
                if len(failed_checks) > len(checks) / 2:
                    status = "unhealthy"
                else:
                    status = "degraded"
                details["failed_checks"] = failed_checks
            
        except Exception as e:
            status = "unhealthy"
            checks["exception"] = False
            details["error"] = str(e)
            logger.error(f"Health check failed for {component}: {e}")
        
        response_time = time.time() - start_time
        
        health_check = HealthCheck(
            component=component,
            status=status,
            checks=checks,
            response_time=response_time,
            timestamp=datetime.now(),
            details=details
        )
        
        # Store health status
        self.health_status[component] = health_check
        
        return health_check
    
    def get_metrics_summary(self, time_window: str = "1h") -> Dict[str, Any]:
        """Get aggregated metrics summary"""
        if time_window not in self.rolling_windows:
            raise ValueError(f"Invalid time window: {time_window}")
        
        window_metrics = list(self.rolling_windows[time_window])
        if not window_metrics:
            return {"message": "No metrics available"}
        
        # Group metrics by component and type
        grouped_metrics = defaultdict(lambda: defaultdict(list))
        for metric in window_metrics:
            grouped_metrics[metric.component][metric.metric_name].append(metric.value)
        
        summary = {}
        for component, metrics in grouped_metrics.items():
            component_summary = {}
            for metric_name, values in metrics.items():
                if values:
                    component_summary[metric_name] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "std": np.std(values),
                        "p95": np.percentile(values, 95),
                        "p99": np.percentile(values, 99)
                    }
            summary[component] = component_summary
        
        return {
            "time_window": time_window,
            "metrics_count": len(window_metrics),
            "components": summary,
            "generated_at": datetime.now().isoformat()
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        # System overview
        system_health = self.check_component_health("system")
        
        # API performance
        api_metrics = self._get_api_performance_summary()
        
        # Model performance
        model_metrics = self._get_model_performance_summary()
        
        # Active alerts
        active_alerts = self.get_active_alerts()
        
        # Recent metrics
        recent_metrics = self.get_metrics_summary("15m")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": {
                "status": system_health.status,
                "cpu_usage": system_health.details.get("cpu_usage", 0),
                "memory_usage": system_health.details.get("memory_usage", 0),
                "disk_usage": system_health.details.get("disk_usage", 0)
            },
            "api_performance": api_metrics,
            "model_performance": model_metrics,
            "active_alerts": len(active_alerts),
            "alert_summary": self._summarize_alerts(active_alerts),
            "recent_metrics": recent_metrics,
            "health_status": {
                component: health.status 
                for component, health in self.health_status.items()
            }
        }
    
    def detect_anomalies(self, component: str, metric_name: str, 
                        window_size: int = 100) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics using statistical methods"""
        # Get recent metrics for the component
        component_metrics = [
            m for m in self.metrics_buffer 
            if m.component == component and m.metric_name == metric_name
        ][-window_size:]
        
        if len(component_metrics) < 10:
            return []
        
        values = [m.value for m in component_metrics]
        timestamps = [m.timestamp for m in component_metrics]
        
        # Calculate statistical thresholds
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Z-score based anomaly detection
        anomalies = []
        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
            
            if z_score > 3:  # 3 standard deviations
                anomalies.append({
                    "timestamp": timestamp.isoformat(),
                    "value": value,
                    "z_score": z_score,
                    "severity": "high" if z_score > 4 else "medium",
                    "expected_range": [mean_val - 2*std_val, mean_val + 2*std_val]
                })
        
        return anomalies
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Record system metrics
                self.record_system_metrics()
                
                # Check component health
                for component in ["system", "api.scoring", "model.insurance_scoring"]:
                    self.check_component_health(component)
                
                # Clean old metrics
                self._cleanup_old_metrics()
                
                # Sleep for monitoring interval
                time.sleep(self.config.get('monitoring_interval', 30))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds and create alerts"""
        threshold_key = f"{metric.component}.{metric.metric_name}"
        
        if threshold_key not in self.thresholds:
            return
        
        threshold_config = self.thresholds[threshold_key]
        
        for severity, threshold_value in threshold_config.items():
            if metric.value > threshold_value:
                alert_id = f"{threshold_key}_{severity}_{int(time.time())}"
                
                alert = Alert(
                    alert_id=alert_id,
                    severity=AlertSeverity(severity.upper()),
                    component=metric.component,
                    metric_name=metric.metric_name,
                    message=f"{metric.metric_name} exceeded {severity} threshold: {metric.value:.2f} > {threshold_value}",
                    value=metric.value,
                    threshold=threshold_value,
                    timestamp=metric.timestamp
                )
                
                self.alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                logger.warning(f"Alert created: {alert.message}")
                break
    
    def _get_api_performance_summary(self) -> Dict[str, Any]:
        """Get API performance summary"""
        api_metrics = [m for m in self.metrics_buffer if m.component.startswith("api.")]
        
        if not api_metrics:
            return {"message": "No API metrics available"}
        
        # Group by endpoint
        endpoint_metrics = defaultdict(lambda: defaultdict(list))
        for metric in api_metrics[-1000:]:  # Last 1000 metrics
            endpoint = metric.component.split(".", 1)[1]
            endpoint_metrics[endpoint][metric.metric_name].append(metric.value)
        
        summary = {}
        for endpoint, metrics in endpoint_metrics.items():
            endpoint_summary = {}
            
            if "response_time" in metrics:
                response_times = metrics["response_time"]
                endpoint_summary["avg_response_time"] = np.mean(response_times)
                endpoint_summary["p95_response_time"] = np.percentile(response_times, 95)
            
            if "error_rate" in metrics:
                errors = metrics["error_rate"]
                endpoint_summary["error_rate"] = np.mean(errors) * 100  # Percentage
            
            if "requests_per_second" in metrics:
                endpoint_summary["throughput"] = len(metrics["requests_per_second"])
            
            summary[endpoint] = endpoint_summary
        
        return summary
    
    def _get_model_performance_summary(self) -> Dict[str, Any]:
        """Get model performance summary"""
        model_metrics = [m for m in self.metrics_buffer if m.component.startswith("model.")]
        
        if not model_metrics:
            return {"message": "No model metrics available"}
        
        # Group by model
        model_summary = defaultdict(lambda: defaultdict(list))
        for metric in model_metrics[-1000:]:
            model_name = metric.component.split(".", 1)[1]
            model_summary[model_name][metric.metric_name].append(metric.value)
        
        summary = {}
        for model_name, metrics in model_summary.items():
            model_stats = {}
            
            if "prediction_time" in metrics:
                pred_times = metrics["prediction_time"]
                model_stats["avg_prediction_time"] = np.mean(pred_times)
                model_stats["p95_prediction_time"] = np.percentile(pred_times, 95)
            
            if "prediction_confidence" in metrics:
                confidences = metrics["prediction_confidence"]
                model_stats["avg_confidence"] = np.mean(confidences)
                model_stats["low_confidence_rate"] = np.mean([c < 0.7 for c in confidences]) * 100
            
            # Get latest accuracy metrics
            accuracy_metrics = [m for m in model_metrics if m.component == f"model.{model_name}" and "accuracy" in m.metric_name]
            if accuracy_metrics:
                latest_accuracy = max(accuracy_metrics, key=lambda x: x.timestamp)
                model_stats["latest_accuracy"] = latest_accuracy.value
                model_stats["accuracy_timestamp"] = latest_accuracy.timestamp.isoformat()
            
            summary[model_name] = model_stats
        
        return summary
    
    def _summarize_alerts(self, alerts: List[Alert]) -> Dict[str, int]:
        """Summarize alerts by severity"""
        summary = defaultdict(int)
        for alert in alerts:
            summary[alert.severity.value] += 1
        return dict(summary)
    
    def _check_prediction_latency(self, model_name: str) -> bool:
        """Check if model prediction latency is within acceptable range"""
        recent_metrics = [
            m for m in self.metrics_buffer 
            if (m.component == f"model.{model_name}" and 
                m.metric_name == "prediction_time" and
                m.timestamp > datetime.now() - timedelta(minutes=5))
        ]
        
        if not recent_metrics:
            return True  # No recent data, assume healthy
        
        avg_latency = np.mean([m.value for m in recent_metrics])
        return avg_latency < 0.1  # 100ms threshold
    
    def _check_accuracy_threshold(self, model_name: str) -> bool:
        """Check if model accuracy is above threshold"""
        recent_accuracy = [
            m for m in self.metrics_buffer 
            if (m.component == f"model.{model_name}" and 
                "accuracy" in m.metric_name and
                m.timestamp > datetime.now() - timedelta(hours=1))
        ]
        
        if not recent_accuracy:
            return True  # No recent data, assume healthy
        
        latest_accuracy = max(recent_accuracy, key=lambda x: x.timestamp)
        return latest_accuracy.value > 0.8  # 80% accuracy threshold
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean metrics buffer
        while self.metrics_buffer and self.metrics_buffer[0].timestamp < cutoff_time:
            self.metrics_buffer.popleft()
        
        # Clean alert history
        while self.alert_history and self.alert_history[0].timestamp < cutoff_time:
            self.alert_history.popleft()
    
    def _update_model_performance(self, metric: PerformanceMetric):
        """Update model performance tracking"""
        model_name = metric.component.split(".", 1)[1]
        self.model_performance[model_name][metric.metric_name].append({
            "value": metric.value,
            "timestamp": metric.timestamp
        })
        
        # Keep only recent data
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.model_performance[model_name][metric.metric_name] = [
            entry for entry in self.model_performance[model_name][metric.metric_name]
            if entry["timestamp"] > cutoff_time
        ]
    
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration"""
        return {
            "monitoring_interval": 30,  # seconds
            "thresholds": {
                "api.scoring.response_time": {"warning": 0.5, "critical": 1.0},
                "api.scoring.error_rate": {"warning": 0.05, "critical": 0.1},
                "model.insurance_scoring.prediction_time": {"warning": 0.1, "critical": 0.2},
                "system.cpu_usage": {"warning": 70, "critical": 85},
                "system.memory_usage": {"warning": 80, "critical": 90},
                "system.disk_usage": {"warning": 85, "critical": 95}
            },
            "alert_channels": ["email", "slack"],
            "retention_hours": 24
        }

# Performance monitoring decorator
def monitor_performance(component: str, metric_name: str = None):
    """Decorator to automatically monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Record performance metric
                if hasattr(wrapper, '_monitor'):
                    metric = PerformanceMetric(
                        metric_name=metric_name or f"{func.__name__}_time",
                        metric_type=MetricType.LATENCY,
                        value=execution_time,
                        timestamp=datetime.now(),
                        component=component,
                        metadata={
                            "function": func.__name__,
                            "success": success,
                            "error": error
                        }
                    )
                    wrapper._monitor.record_metric(metric)
            
            return result
        
        return wrapper
    return decorator

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Context manager for monitoring operations
class MonitoredOperation:
    """Context manager for monitoring operations"""
    
    def __init__(self, component: str, operation_name: str, 
                 monitor: PerformanceMonitor = None):
        self.component = component
        self.operation_name = operation_name
        self.monitor = monitor or performance_monitor
        self.start_time = None
        self.metadata = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        # Record the operation
        metric = PerformanceMetric(
            metric_name=f"{self.operation_name}_time",
            metric_type=MetricType.LATENCY,
            value=execution_time,
            timestamp=datetime.now(),
            component=self.component,
            metadata={
                "operation": self.operation_name,
                "success": exc_type is None,
                "error": str(exc_val) if exc_val else None,
                **self.metadata
            }
        )
        
        self.monitor.record_metric(metric)
    
    def add_metadata(self, **kwargs):
        """Add metadata to the operation"""
        self.metadata.update(kwargs)

if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Simulate some metrics
    import time
    import random
    
    for i in range(100):
        # Simulate API calls
        response_time = random.uniform(0.05, 0.3)
        status_code = random.choice([200, 200, 200, 400, 500])
        
        monitor.record_api_call(
            endpoint="score-lead",
            method="POST",
            status_code=status_code,
            response_time=response_time
        )
        
        # Simulate model predictions
        prediction_time = random.uniform(0.01, 0.1)
        confidence = random.uniform(0.6, 0.95)
        
        monitor.record_model_prediction(
            model_name="insurance_scoring",
            prediction_time=prediction_time,
            input_features={"age": 35, "income": 50000},
            prediction=0.75,
            confidence=confidence
        )
        
        time.sleep(0.1)
    
    # Get dashboard
    dashboard = monitor.get_performance_dashboard()
    print("Performance Dashboard:")
    print(json.dumps(dashboard, indent=2, default=str))
    
    # Check for anomalies
    anomalies = monitor.detect_anomalies("api.score-lead", "response_time")
    if anomalies:
        print(f"\nDetected {len(anomalies)} anomalies in API response time")
    
    monitor.stop_monitoring()