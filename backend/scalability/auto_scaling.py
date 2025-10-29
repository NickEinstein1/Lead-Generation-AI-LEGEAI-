import asyncio
import psutil
import docker
import kubernetes
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import time
import statistics
from collections import deque
import os

logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down"
    NONE = "none"

class ScalingTrigger(Enum):
    """Scaling trigger types"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"

@dataclass
class ScalingMetric:
    """Scaling metric definition"""
    name: str
    trigger_type: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    evaluation_window: int = 300  # 5 minutes
    cooldown_period: int = 300    # 5 minutes
    weight: float = 1.0
    enabled: bool = True

@dataclass
class ScalingRule:
    """Scaling rule configuration"""
    name: str
    metrics: List[ScalingMetric]
    min_instances: int = 1
    max_instances: int = 10
    scale_up_step: int = 1
    scale_down_step: int = 1
    target_service: str = ""
    enabled: bool = True

@dataclass
class ScalingEvent:
    """Scaling event record"""
    timestamp: datetime
    service: str
    direction: ScalingDirection
    from_instances: int
    to_instances: int
    trigger_metric: str
    trigger_value: float
    reason: str

class MetricsCollector:
    """Collects system and application metrics"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics_history: Dict[str, deque] = {}
        self.running = False
        
        # Initialize metric queues
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize metric storage"""
        metric_names = [
            'cpu_usage', 'memory_usage', 'disk_usage',
            'queue_length', 'response_time', 'error_rate',
            'active_connections', 'throughput'
        ]
        
        for metric in metric_names:
            self.metrics_history[metric] = deque(maxlen=1440)  # 24 hours at 1-minute intervals
    
    async def start_collection(self):
        """Start metrics collection"""
        self.running = True
        
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect application metrics
                await self._collect_application_metrics()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            self._record_metric('cpu_usage', cpu_usage)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            self._record_metric('memory_usage', memory_usage)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            self._record_metric('disk_usage', disk_usage)
            
            logger.debug(f"System metrics - CPU: {cpu_usage}%, Memory: {memory_usage}%, Disk: {disk_usage}%")
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            from backend.scalability.distributed_processing import task_queue, worker_manager
            
            # Queue metrics
            queue_stats = task_queue.get_queue_stats()
            total_pending = queue_stats.get('total_pending', 0)
            self._record_metric('queue_length', total_pending)
            
            # Worker metrics
            worker_stats = worker_manager.get_worker_stats()
            active_workers = worker_stats.get('total_workers', 0)
            self._record_metric('active_workers', active_workers)
            
            # Performance metrics (would come from monitoring system)
            # For now, simulate some metrics
            import random
            response_time = random.uniform(0.1, 2.0)  # Simulated
            error_rate = random.uniform(0, 0.1)       # Simulated
            
            self._record_metric('response_time', response_time)
            self._record_metric('error_rate', error_rate)
            
            logger.debug(f"App metrics - Queue: {total_pending}, Workers: {active_workers}")
            
        except Exception as e:
            logger.error(f"Application metrics collection error: {e}")
    
    def _record_metric(self, metric_name: str, value: float):
        """Record a metric value with timestamp"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=1440)
        
        self.metrics_history[metric_name].append({
            'timestamp': datetime.now(),
            'value': value
        })
    
    def get_metric_average(self, metric_name: str, window_seconds: int = 300) -> Optional[float]:
        """Get average metric value over time window"""
        if metric_name not in self.metrics_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_values = [
            entry['value'] for entry in self.metrics_history[metric_name]
            if entry['timestamp'] >= cutoff_time
        ]
        
        if not recent_values:
            return None
        
        return statistics.mean(recent_values)
    
    def get_metric_trend(self, metric_name: str, window_seconds: int = 300) -> Optional[str]:
        """Get metric trend (increasing, decreasing, stable)"""
        if metric_name not in self.metrics_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_entries = [
            entry for entry in self.metrics_history[metric_name]
            if entry['timestamp'] >= cutoff_time
        ]
        
        if len(recent_entries) < 2:
            return "stable"
        
        # Calculate trend using linear regression slope
        values = [entry['value'] for entry in recent_entries]
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

class ContainerOrchestrator:
    """Abstract base class for container orchestration"""
    
    async def scale_service(self, service_name: str, target_instances: int) -> bool:
        """Scale a service to target number of instances"""
        raise NotImplementedError
    
    async def get_service_instances(self, service_name: str) -> int:
        """Get current number of service instances"""
        raise NotImplementedError
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get service status information"""
        raise NotImplementedError

class DockerOrchestrator(ContainerOrchestrator):
    """Docker-based container orchestration"""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.client = None
    
    async def scale_service(self, service_name: str, target_instances: int) -> bool:
        """Scale Docker service"""
        if not self.client:
            return False
        
        try:
            # For Docker Swarm services
            service = self.client.services.get(service_name)
            service.update(mode={'Replicated': {'Replicas': target_instances}})
            
            logger.info(f"Scaled Docker service {service_name} to {target_instances} instances")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale Docker service {service_name}: {e}")
            return False
    
    async def get_service_instances(self, service_name: str) -> int:
        """Get current Docker service instances"""
        if not self.client:
            return 0
        
        try:
            service = self.client.services.get(service_name)
            spec = service.attrs['Spec']
            return spec.get('Mode', {}).get('Replicated', {}).get('Replicas', 0)
            
        except Exception as e:
            logger.error(f"Failed to get Docker service instances for {service_name}: {e}")
            return 0
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get Docker service status"""
        if not self.client:
            return {}
        
        try:
            service = self.client.services.get(service_name)
            tasks = service.tasks()
            
            running_tasks = sum(1 for task in tasks if task['Status']['State'] == 'running')
            
            return {
                'service_name': service_name,
                'desired_instances': await self.get_service_instances(service_name),
                'running_instances': running_tasks,
                'status': 'healthy' if running_tasks > 0 else 'unhealthy'
            }
            
        except Exception as e:
            logger.error(f"Failed to get Docker service status for {service_name}: {e}")
            return {}

class KubernetesOrchestrator(ContainerOrchestrator):
    """Kubernetes-based container orchestration"""
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        try:
            # Try to load in-cluster config first, then local config
            try:
                kubernetes.config.load_incluster_config()
            except:
                kubernetes.config.load_kube_config()
            
            self.apps_v1 = kubernetes.client.AppsV1Api()
            self.core_v1 = kubernetes.client.CoreV1Api()
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            self.apps_v1 = None
            self.core_v1 = None
    
    async def scale_service(self, service_name: str, target_instances: int) -> bool:
        """Scale Kubernetes deployment"""
        if not self.apps_v1:
            return False
        
        try:
            # Update deployment replicas
            body = {'spec': {'replicas': target_instances}}
            
            self.apps_v1.patch_namespaced_deployment_scale(
                name=service_name,
                namespace=self.namespace,
                body=body
            )
            
            logger.info(f"Scaled Kubernetes deployment {service_name} to {target_instances} instances")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale Kubernetes deployment {service_name}: {e}")
            return False
    
    async def get_service_instances(self, service_name: str) -> int:
        """Get current Kubernetes deployment instances"""
        if not self.apps_v1:
            return 0
        
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace=self.namespace
            )
            return deployment.spec.replicas or 0
            
        except Exception as e:
            logger.error(f"Failed to get Kubernetes deployment instances for {service_name}: {e}")
            return 0
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get Kubernetes deployment status"""
        if not self.apps_v1:
            return {}
        
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace=self.namespace
            )
            
            desired = deployment.spec.replicas or 0
            ready = deployment.status.ready_replicas or 0
            available = deployment.status.available_replicas or 0
            
            return {
                'service_name': service_name,
                'desired_instances': desired,
                'ready_instances': ready,
                'available_instances': available,
                'status': 'healthy' if ready == desired else 'scaling'
            }
            
        except Exception as e:
            logger.error(f"Failed to get Kubernetes deployment status for {service_name}: {e}")
            return {}

class AutoScaler:
    """Intelligent auto-scaling system"""
    
    def __init__(self, orchestrator: ContainerOrchestrator, 
                 metrics_collector: MetricsCollector):
        self.orchestrator = orchestrator
        self.metrics_collector = metrics_collector
        
        # Scaling configuration
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_history: deque = deque(maxlen=1000)
        self.last_scaling_action: Dict[str, datetime] = {}
        
        # State
        self.running = False
        self.evaluation_interval = 60  # Check every minute
        
        # Load default scaling rules
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default scaling rules"""
        
        # API service scaling rule
        api_metrics = [
            ScalingMetric(
                name="cpu_usage",
                trigger_type=ScalingTrigger.CPU_USAGE,
                scale_up_threshold=70.0,
                scale_down_threshold=30.0,
                weight=1.0
            ),
            ScalingMetric(
                name="memory_usage", 
                trigger_type=ScalingTrigger.MEMORY_USAGE,
                scale_up_threshold=80.0,
                scale_down_threshold=40.0,
                weight=0.8
            ),
            ScalingMetric(
                name="response_time",
                trigger_type=ScalingTrigger.RESPONSE_TIME,
                scale_up_threshold=1.0,  # 1 second
                scale_down_threshold=0.3,  # 300ms
                weight=1.2
            )
        ]
        
        self.scaling_rules["api-service"] = ScalingRule(
            name="api-service",
            metrics=api_metrics,
            min_instances=2,
            max_instances=20,
            scale_up_step=2,
            scale_down_step=1,
            target_service="insurance-scoring-api"
        )
        
        # Worker scaling rule
        worker_metrics = [
            ScalingMetric(
                name="queue_length",
                trigger_type=ScalingTrigger.QUEUE_LENGTH,
                scale_up_threshold=50.0,
                scale_down_threshold=10.0,
                weight=1.5
            ),
            ScalingMetric(
                name="cpu_usage",
                trigger_type=ScalingTrigger.CPU_USAGE,
                scale_up_threshold=75.0,
                scale_down_threshold=25.0,
                weight=1.0
            )
        ]
        
        self.scaling_rules["worker-service"] = ScalingRule(
            name="worker-service",
            metrics=worker_metrics,
            min_instances=1,
            max_instances=50,
            scale_up_step=3,
            scale_down_step=2,
            target_service="lead-processing-workers"
        )
    
    async def start(self):
        """Start the auto-scaling system"""
        self.running = True
        logger.info("Auto-scaler started")
        
        while self.running:
            try:
                # Evaluate all scaling rules
                for rule_name, rule in self.scaling_rules.items():
                    if rule.enabled:
                        await self._evaluate_scaling_rule(rule)
                
                await asyncio.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Auto-scaling evaluation error: {e}")
                await asyncio.sleep(10)
    
    async def stop(self):
        """Stop the auto-scaling system"""
        self.running = False
        logger.info("Auto-scaler stopped")
    
    async def _evaluate_scaling_rule(self, rule: ScalingRule):
        """Evaluate a scaling rule and take action if needed"""
        try:
            # Check cooldown period
            if self._is_in_cooldown(rule.name):
                return
            
            # Get current instances
            current_instances = await self.orchestrator.get_service_instances(rule.target_service)
            
            # Calculate scaling decision
            scaling_decision = await self._calculate_scaling_decision(rule)
            
            if scaling_decision['direction'] == ScalingDirection.NONE:
                return
            
            # Calculate target instances
            if scaling_decision['direction'] == ScalingDirection.UP:
                target_instances = min(
                    current_instances + rule.scale_up_step,
                    rule.max_instances
                )
            else:  # Scale down
                target_instances = max(
                    current_instances - rule.scale_down_step,
                    rule.min_instances
                )
            
            # Perform scaling if needed
            if target_instances != current_instances:
                success = await self.orchestrator.scale_service(
                    rule.target_service, 
                    target_instances
                )
                
                if success:
                    # Record scaling event
                    event = ScalingEvent(
                        timestamp=datetime.now(),
                        service=rule.target_service,
                        direction=scaling_decision['direction'],
                        from_instances=current_instances,
                        to_instances=target_instances,
                        trigger_metric=scaling_decision['trigger_metric'],
                        trigger_value=scaling_decision['trigger_value'],
                        reason=scaling_decision['reason']
                    )
                    
                    self.scaling_history.append(event)
                    self.last_scaling_action[rule.name] = datetime.now()
                    
                    logger.info(f"Scaled {rule.target_service} from {current_instances} to {target_instances} instances")
                    logger.info(f"Trigger: {scaling_decision['trigger_metric']} = {scaling_decision['trigger_value']}")
                
        except Exception as e:
            logger.error(f"Failed to evaluate scaling rule {rule.name}: {e}")
    
    async def _calculate_scaling_decision(self, rule: ScalingRule) -> Dict[str, Any]:
        """Calculate scaling decision based on metrics"""
        scale_up_score = 0.0
        scale_down_score = 0.0
        trigger_metric = ""
        trigger_value = 0.0
        
        for metric in rule.metrics:
            if not metric.enabled:
                continue
            
            # Get metric value
            avg_value = self.metrics_collector.get_metric_average(
                metric.name, 
                metric.evaluation_window
            )
            
            if avg_value is None:
                continue
            
            # Calculate scaling scores
            if avg_value > metric.scale_up_threshold:
                score = ((avg_value - metric.scale_up_threshold) / 
                        metric.scale_up_threshold) * metric.weight
                scale_up_score += score
                
                if score > 0.5:  # Significant trigger
                    trigger_metric = metric.name
                    trigger_value = avg_value
            
            elif avg_value < metric.scale_down_threshold:
                score = ((metric.scale_down_threshold - avg_value) / 
                        metric.scale_down_threshold) * metric.weight
                scale_down_score += score
                
                if score > 0.5:  # Significant trigger
                    trigger_metric = metric.name
                    trigger_value = avg_value
        
        # Make scaling decision
        if scale_up_score > scale_down_score and scale_up_score > 1.0:
            return {
                'direction': ScalingDirection.UP,
                'trigger_metric': trigger_metric,
                'trigger_value': trigger_value,
                'reason': f"Scale up triggered by {trigger_metric} = {trigger_value:.2f}"
            }
        elif scale_down_score > scale_up_score and scale_down_score > 1.0:
            return {
                'direction': ScalingDirection.DOWN,
                'trigger_metric': trigger_metric,
                'trigger_value': trigger_value,
                'reason': f"Scale down triggered by {trigger_metric} = {trigger_value:.2f}"
            }
        else:
            return {
                'direction': ScalingDirection.NONE,
                'trigger_metric': '',
                'trigger_value': 0.0,
                'reason': 'No scaling needed'
            }
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if rule is in cooldown period"""
        if rule_name not in self.last_scaling_action:
            return False
        
        rule = self.scaling_rules[rule_name]
        last_action = self.last_scaling_action[rule_name]
        cooldown_end = last_action + timedelta(seconds=rule.metrics[0].cooldown_period)
        
        return datetime.now() < cooldown_end
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule"""
        self.scaling_rules[rule.name] = rule
        logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule"""
        if rule_name in self.scaling_rules:
            del self.scaling_rules[rule_name]
            logger.info(f"Removed scaling rule: {rule_name}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        status = {
            'active_rules': len([r for r in self.scaling_rules.values() if r.enabled]),
            'total_rules': len(self.scaling_rules),
            'recent_events': [],
            'service_status': {}
        }
        
        # Recent scaling events (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_events = [
            {
                'timestamp': event.timestamp.isoformat(),
                'service': event.service,
                'direction': event.direction.value,
                'from_instances': event.from_instances,
                'to_instances': event.to_instances,
                'trigger': event.trigger_metric,
                'reason': event.reason
            }
            for event in self.scaling_history
            if event.timestamp >= cutoff_time
        ]
        
        status['recent_events'] = recent_events[-10:]  # Last 10 events
        
        return status

# Global instances
metrics_collector = MetricsCollector()

# Choose orchestrator based on environment
if os.getenv('KUBERNETES_SERVICE_HOST'):
    orchestrator = KubernetesOrchestrator()
else:
    orchestrator = DockerOrchestrator()

auto_scaler = AutoScaler(orchestrator, metrics_collector)

# Utility functions
async def start_auto_scaling():
    """Start the auto-scaling system"""
    # Start metrics collection
    metrics_task = asyncio.create_task(metrics_collector.start_collection())
    
    # Start auto-scaler
    scaler_task = asyncio.create_task(auto_scaler.start())
    
    return metrics_task, scaler_task

async def stop_auto_scaling(metrics_task, scaler_task):
    """Stop the auto-scaling system"""
    await auto_scaler.stop()
    metrics_collector.running = False
    
    metrics_task.cancel()
    scaler_task.cancel()

# Example usage
async def example_usage():
    """Example of using the auto-scaling system"""
    
    # Start auto-scaling
    metrics_task, scaler_task = await start_auto_scaling()
    
    try:
        # Run for a while
        await asyncio.sleep(300)  # 5 minutes
        
        # Get status
        status = auto_scaler.get_scaling_status()
        print("Auto-scaling Status:", json.dumps(status, indent=2, default=str))
        
    finally:
        # Stop auto-scaling
        await stop_auto_scaling(metrics_task, scaler_task)

if __name__ == "__main__":
    asyncio.run(example_usage())
