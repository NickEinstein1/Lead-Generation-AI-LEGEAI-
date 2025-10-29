import asyncio
import redis
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
import threading
from abc import ABC, abstractmethod
import psutil
import os

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5

@dataclass
class Task:
    """Distributed task definition"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_count: int = 0
    timeout: int = 300  # 5 minutes
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    worker_id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkerStats:
    """Worker performance statistics"""
    worker_id: str
    tasks_processed: int = 0
    tasks_failed: int = 0
    avg_processing_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    status: str = "idle"

class TaskQueue:
    """Distributed task queue with Redis backend"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'decode_responses': False
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        self.queue_name = "lead_processing_queue"
        self.result_queue = "lead_processing_results"
        self.worker_registry = "active_workers"
        
        # Queue names by priority
        self.priority_queues = {
            TaskPriority.URGENT: f"{self.queue_name}:urgent",
            TaskPriority.CRITICAL: f"{self.queue_name}:critical", 
            TaskPriority.HIGH: f"{self.queue_name}:high",
            TaskPriority.NORMAL: f"{self.queue_name}:normal",
            TaskPriority.LOW: f"{self.queue_name}:low"
        }
    
    def enqueue_task(self, task: Task) -> str:
        """Add task to appropriate priority queue"""
        try:
            # Serialize task
            task_data = pickle.dumps(task)
            
            # Add to priority queue
            queue_name = self.priority_queues[task.priority]
            self.redis_client.lpush(queue_name, task_data)
            
            # Store task metadata for tracking
            self.redis_client.hset(
                f"task:{task.task_id}",
                mapping={
                    'status': task.status.value,
                    'created_at': task.created_at.isoformat(),
                    'priority': task.priority.value,
                    'task_type': task.task_type
                }
            )
            
            # Set expiration for task metadata (24 hours)
            self.redis_client.expire(f"task:{task.task_id}", 86400)
            
            logger.info(f"Task {task.task_id} enqueued with priority {task.priority.name}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            raise
    
    def dequeue_task(self, worker_id: str, timeout: int = 10) -> Optional[Task]:
        """Dequeue highest priority task"""
        try:
            # Check queues in priority order
            for priority in [TaskPriority.URGENT, TaskPriority.CRITICAL, 
                           TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
                queue_name = self.priority_queues[priority]
                
                # Blocking pop with timeout
                result = self.redis_client.brpop(queue_name, timeout=timeout)
                
                if result:
                    _, task_data = result
                    task = pickle.loads(task_data)
                    
                    # Update task status
                    task.status = TaskStatus.PROCESSING
                    task.started_at = datetime.now()
                    task.worker_id = worker_id
                    
                    # Update task metadata
                    self.redis_client.hset(
                        f"task:{task.task_id}",
                        mapping={
                            'status': task.status.value,
                            'started_at': task.started_at.isoformat(),
                            'worker_id': worker_id
                        }
                    )
                    
                    logger.info(f"Task {task.task_id} dequeued by worker {worker_id}")
                    return task
            
            return None  # No tasks available
            
        except Exception as e:
            logger.error(f"Failed to dequeue task for worker {worker_id}: {e}")
            return None
    
    def complete_task(self, task: Task, result: Any = None):
        """Mark task as completed"""
        try:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Update task metadata
            self.redis_client.hset(
                f"task:{task.task_id}",
                mapping={
                    'status': task.status.value,
                    'completed_at': task.completed_at.isoformat()
                }
            )
            
            # Store result
            if result is not None:
                result_data = {
                    'task_id': task.task_id,
                    'result': pickle.dumps(result),
                    'completed_at': task.completed_at.isoformat()
                }
                self.redis_client.lpush(self.result_queue, pickle.dumps(result_data))
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to complete task {task.task_id}: {e}")
            raise
    
    def fail_task(self, task: Task, error: str):
        """Mark task as failed and handle retries"""
        try:
            task.retry_count += 1
            task.error = error
            
            if task.retry_count <= task.max_retries:
                # Retry the task
                task.status = TaskStatus.RETRYING
                task.started_at = None
                task.worker_id = None
                
                # Re-enqueue with exponential backoff
                delay = min(300, 2 ** task.retry_count)  # Max 5 minutes
                
                # Schedule retry (simplified - in production use Redis delayed queues)
                self.enqueue_task(task)
                
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
            else:
                # Max retries exceeded
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                
                logger.error(f"Task {task.task_id} failed permanently after {task.retry_count} retries: {error}")
            
            # Update task metadata
            self.redis_client.hset(
                f"task:{task.task_id}",
                mapping={
                    'status': task.status.value,
                    'retry_count': task.retry_count,
                    'error': error
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to handle task failure for {task.task_id}: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and metadata"""
        try:
            task_data = self.redis_client.hgetall(f"task:{task_id}")
            if task_data:
                return {k.decode(): v.decode() for k, v in task_data.items()}
            return None
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            stats = {}
            total_pending = 0
            
            for priority, queue_name in self.priority_queues.items():
                queue_length = self.redis_client.llen(queue_name)
                stats[priority.name.lower()] = queue_length
                total_pending += queue_length
            
            stats['total_pending'] = total_pending
            stats['active_workers'] = self.redis_client.scard(self.worker_registry)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}

class TaskProcessor(ABC):
    """Abstract base class for task processors"""
    
    @abstractmethod
    async def process(self, task: Task) -> Any:
        """Process a task and return result"""
        pass
    
    @abstractmethod
    def get_supported_task_types(self) -> List[str]:
        """Return list of supported task types"""
        pass

class LeadScoringProcessor(TaskProcessor):
    """Processor for lead scoring tasks"""
    
    def __init__(self):
        from backend.models.meta_lead_generation.inference import MetaLeadGenerationInference
        self.scorer = MetaLeadGenerationInference()
    
    async def process(self, task: Task) -> Any:
        """Process lead scoring task"""
        try:
            lead_data = task.payload.get('lead_data')
            if not lead_data:
                raise ValueError("Missing lead_data in task payload")
            
            # Score the lead
            if task.payload.get('with_enrichment', False):
                result = await self.scorer.score_lead_with_enrichment(lead_data)
            else:
                result = self.scorer.score_lead(lead_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Lead scoring failed for task {task.task_id}: {e}")
            raise
    
    def get_supported_task_types(self) -> List[str]:
        return ['lead_scoring', 'lead_scoring_with_enrichment']

class BatchLeadProcessor(TaskProcessor):
    """Processor for batch lead operations"""
    
    def __init__(self):
        from backend.models.meta_lead_generation.inference import MetaLeadGenerationInference
        self.scorer = MetaLeadGenerationInference()
    
    async def process(self, task: Task) -> Any:
        """Process batch lead scoring"""
        try:
            leads_data = task.payload.get('leads_data')
            if not leads_data:
                raise ValueError("Missing leads_data in task payload")
            
            batch_size = task.payload.get('batch_size', 50)
            results = []
            
            # Process in chunks
            for i in range(0, len(leads_data), batch_size):
                chunk = leads_data[i:i + batch_size]
                chunk_results = []
                
                for lead_data in chunk:
                    try:
                        if task.payload.get('with_enrichment', False):
                            result = await self.scorer.score_lead_with_enrichment(lead_data)
                        else:
                            result = self.scorer.score_lead(lead_data)
                        chunk_results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to score lead {lead_data.get('lead_id', 'unknown')}: {e}")
                        chunk_results.append({
                            'lead_id': lead_data.get('lead_id', 'unknown'),
                            'error': str(e),
                            'score': 0
                        })
                
                results.extend(chunk_results)
                
                # Small delay between chunks to prevent overwhelming
                await asyncio.sleep(0.1)
            
            return {
                'total_leads': len(leads_data),
                'processed_leads': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed for task {task.task_id}: {e}")
            raise
    
    def get_supported_task_types(self) -> List[str]:
        return ['batch_lead_scoring', 'batch_lead_enrichment']

class DataEnrichmentProcessor(TaskProcessor):
    """Processor for data enrichment tasks"""
    
    def __init__(self):
        from backend.models.integrations.api_manager import integration_manager
        self.integration_manager = integration_manager
    
    async def process(self, task: Task) -> Any:
        """Process data enrichment task"""
        try:
            lead_data = task.payload.get('lead_data')
            sources = task.payload.get('sources')
            
            if not lead_data:
                raise ValueError("Missing lead_data in task payload")
            
            enriched_data = await self.integration_manager.enrich_lead_data(
                lead_data=lead_data,
                sources=sources
            )
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Data enrichment failed for task {task.task_id}: {e}")
            raise
    
    def get_supported_task_types(self) -> List[str]:
        return ['data_enrichment', 'lead_enrichment']

class DistributedWorker:
    """Distributed worker for processing tasks"""
    
    def __init__(self, worker_id: str = None, task_queue: TaskQueue = None,
                 processors: List[TaskProcessor] = None):
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.task_queue = task_queue or TaskQueue()
        self.processors = processors or [
            LeadScoringProcessor(),
            BatchLeadProcessor(),
            DataEnrichmentProcessor()
        ]
        
        # Build processor mapping
        self.processor_map = {}
        for processor in self.processors:
            for task_type in processor.get_supported_task_types():
                self.processor_map[task_type] = processor
        
        # Worker state
        self.running = False
        self.stats = WorkerStats(worker_id=self.worker_id)
        self.current_task: Optional[Task] = None
        
        # Performance monitoring
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def start(self):
        """Start the worker"""
        self.running = True
        
        # Register worker
        self.task_queue.redis_client.sadd(
            self.task_queue.worker_registry,
            self.worker_id
        )
        
        logger.info(f"Worker {self.worker_id} started")
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start processing loop
        try:
            await self._processing_loop()
        finally:
            heartbeat_task.cancel()
            await self._cleanup()
    
    async def stop(self):
        """Stop the worker gracefully"""
        self.running = False
        logger.info(f"Worker {self.worker_id} stopping...")
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get next task
                task = self.task_queue.dequeue_task(self.worker_id, timeout=5)
                
                if task:
                    self.current_task = task
                    self.stats.status = "processing"
                    
                    # Process task
                    start_time = time.time()
                    
                    try:
                        result = await self._process_task(task)
                        
                        # Task completed successfully
                        self.task_queue.complete_task(task, result)
                        self.stats.tasks_processed += 1
                        
                        # Update average processing time
                        processing_time = time.time() - start_time
                        self._update_avg_processing_time(processing_time)
                        
                        logger.info(f"Worker {self.worker_id} completed task {task.task_id} in {processing_time:.2f}s")
                        
                    except Exception as e:
                        # Task failed
                        self.task_queue.fail_task(task, str(e))
                        self.stats.tasks_failed += 1
                        
                        logger.error(f"Worker {self.worker_id} failed task {task.task_id}: {e}")
                    
                    finally:
                        self.current_task = None
                        self.stats.status = "idle"
                
                else:
                    # No tasks available, short sleep
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Worker {self.worker_id} processing loop error: {e}")
                await asyncio.sleep(5)
    
    async def _process_task(self, task: Task) -> Any:
        """Process a single task"""
        processor = self.processor_map.get(task.task_type)
        
        if not processor:
            raise ValueError(f"No processor found for task type: {task.task_type}")
        
        # Set timeout
        try:
            result = await asyncio.wait_for(
                processor.process(task),
                timeout=task.timeout
            )
            return result
        except asyncio.TimeoutError:
            raise Exception(f"Task {task.task_id} timed out after {task.timeout} seconds")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat"""
        while self.running:
            try:
                # Update system stats
                self.stats.cpu_usage = psutil.cpu_percent()
                self.stats.memory_usage = psutil.virtual_memory().percent
                self.stats.last_heartbeat = datetime.now()
                
                # Store heartbeat in Redis
                heartbeat_data = {
                    'worker_id': self.worker_id,
                    'status': self.stats.status,
                    'tasks_processed': self.stats.tasks_processed,
                    'tasks_failed': self.stats.tasks_failed,
                    'cpu_usage': self.stats.cpu_usage,
                    'memory_usage': self.stats.memory_usage,
                    'last_heartbeat': self.stats.last_heartbeat.isoformat(),
                    'current_task': self.current_task.task_id if self.current_task else None
                }
                
                self.task_queue.redis_client.hset(
                    f"worker:{self.worker_id}",
                    mapping=heartbeat_data
                )
                
                # Set expiration (worker considered dead after 60 seconds)
                self.task_queue.redis_client.expire(f"worker:{self.worker_id}", 60)
                
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error for worker {self.worker_id}: {e}")
                await asyncio.sleep(5)
    
    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time"""
        if self.stats.tasks_processed == 1:
            self.stats.avg_processing_time = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_processing_time = (
                alpha * processing_time + 
                (1 - alpha) * self.stats.avg_processing_time
            )
    
    async def _cleanup(self):
        """Cleanup worker resources"""
        try:
            # Remove from worker registry
            self.task_queue.redis_client.srem(
                self.task_queue.worker_registry,
                self.worker_id
            )
            
            # Remove worker heartbeat
            self.task_queue.redis_client.delete(f"worker:{self.worker_id}")
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info(f"Worker {self.worker_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error for worker {self.worker_id}: {e}")

class WorkerManager:
    """Manages multiple workers and auto-scaling"""
    
    def __init__(self, task_queue: TaskQueue = None, min_workers: int = 2, 
                 max_workers: int = 10):
        self.task_queue = task_queue or TaskQueue()
        self.min_workers = min_workers
        self.max_workers = max_workers
        
        self.workers: Dict[str, DistributedWorker] = {}
        self.worker_tasks: Dict[str, asyncio.Task] = {}
        
        # Auto-scaling configuration
        self.scale_up_threshold = 10  # Queue length to scale up
        self.scale_down_threshold = 2  # Queue length to scale down
        self.scale_check_interval = 30  # Check every 30 seconds
        
        self.running = False
    
    async def start(self):
        """Start the worker manager"""
        self.running = True
        
        # Start minimum workers
        for i in range(self.min_workers):
            await self._start_worker()
        
        # Start auto-scaling loop
        scaling_task = asyncio.create_task(self._auto_scaling_loop())
        
        try:
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
        finally:
            scaling_task.cancel()
            await self._stop_all_workers()
    
    async def stop(self):
        """Stop the worker manager"""
        self.running = False
    
    async def _start_worker(self) -> str:
        """Start a new worker"""
        worker = DistributedWorker(task_queue=self.task_queue)
        worker_task = asyncio.create_task(worker.start())
        
        self.workers[worker.worker_id] = worker
        self.worker_tasks[worker.worker_id] = worker_task
        
        logger.info(f"Started worker {worker.worker_id}")
        return worker.worker_id
    
    async def _stop_worker(self, worker_id: str):
        """Stop a specific worker"""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            await worker.stop()
            
            # Cancel the worker task
            if worker_id in self.worker_tasks:
                self.worker_tasks[worker_id].cancel()
                del self.worker_tasks[worker_id]
            
            del self.workers[worker_id]
            logger.info(f"Stopped worker {worker_id}")
    
    async def _stop_all_workers(self):
        """Stop all workers"""
        worker_ids = list(self.workers.keys())
        for worker_id in worker_ids:
            await self._stop_worker(worker_id)
    
    async def _auto_scaling_loop(self):
        """Auto-scaling loop"""
        while self.running:
            try:
                # Get queue statistics
                stats = self.task_queue.get_queue_stats()
                total_pending = stats.get('total_pending', 0)
                active_workers = len(self.workers)
                
                logger.debug(f"Auto-scaling check: {total_pending} pending tasks, {active_workers} workers")
                
                # Scale up if needed
                if (total_pending > self.scale_up_threshold and 
                    active_workers < self.max_workers):
                    
                    workers_to_add = min(
                        self.max_workers - active_workers,
                        max(1, total_pending // self.scale_up_threshold)
                    )
                    
                    for _ in range(workers_to_add):
                        await self._start_worker()
                    
                    logger.info(f"Scaled up: added {workers_to_add} workers (total: {len(self.workers)})")
                
                # Scale down if needed
                elif (total_pending < self.scale_down_threshold and 
                      active_workers > self.min_workers):
                    
                    workers_to_remove = min(
                        active_workers - self.min_workers,
                        max(1, (self.scale_down_threshold - total_pending))
                    )
                    
                    # Remove idle workers first
                    idle_workers = []
                    for worker_id, worker in self.workers.items():
                        if worker.stats.status == "idle":
                            idle_workers.append(worker_id)
                    
                    workers_to_stop = idle_workers[:workers_to_remove]
                    for worker_id in workers_to_stop:
                        await self._stop_worker(worker_id)
                    
                    if workers_to_stop:
                        logger.info(f"Scaled down: removed {len(workers_to_stop)} workers (total: {len(self.workers)})")
                
                await asyncio.sleep(self.scale_check_interval)
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(10)
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics for all workers"""
        stats = {
            'total_workers': len(self.workers),
            'worker_details': {},
            'aggregate_stats': {
                'total_tasks_processed': 0,
                'total_tasks_failed': 0,
                'avg_cpu_usage': 0,
                'avg_memory_usage': 0
            }
        }
        
        if not self.workers:
            return stats
        
        total_cpu = 0
        total_memory = 0
        
        for worker_id, worker in self.workers.items():
            worker_stats = {
                'worker_id': worker_id,
                'status': worker.stats.status,
                'tasks_processed': worker.stats.tasks_processed,
                'tasks_failed': worker.stats.tasks_failed,
                'avg_processing_time': worker.stats.avg_processing_time,
                'cpu_usage': worker.stats.cpu_usage,
                'memory_usage': worker.stats.memory_usage,
                'last_heartbeat': worker.stats.last_heartbeat.isoformat(),
                'current_task': worker.current_task.task_id if worker.current_task else None
            }
            
            stats['worker_details'][worker_id] = worker_stats
            
            # Aggregate stats
            stats['aggregate_stats']['total_tasks_processed'] += worker.stats.tasks_processed
            stats['aggregate_stats']['total_tasks_failed'] += worker.stats.tasks_failed
            total_cpu += worker.stats.cpu_usage
            total_memory += worker.stats.memory_usage
        
        # Calculate averages
        worker_count = len(self.workers)
        stats['aggregate_stats']['avg_cpu_usage'] = total_cpu / worker_count
        stats['aggregate_stats']['avg_memory_usage'] = total_memory / worker_count
        
        return stats

# Global instances
task_queue = TaskQueue()
worker_manager = WorkerManager(task_queue=task_queue)

# Utility functions
def submit_lead_scoring_task(lead_data: Dict[str, Any], 
                           priority: TaskPriority = TaskPriority.NORMAL,
                           with_enrichment: bool = False) -> str:
    """Submit a lead scoring task"""
    task = Task(
        task_id=f"lead_score_{uuid.uuid4().hex[:8]}",
        task_type="lead_scoring_with_enrichment" if with_enrichment else "lead_scoring",
        payload={
            'lead_data': lead_data,
            'with_enrichment': with_enrichment
        },
        priority=priority
    )
    
    return task_queue.enqueue_task(task)

def submit_batch_scoring_task(leads_data: List[Dict[str, Any]], 
                            priority: TaskPriority = TaskPriority.NORMAL,
                            batch_size: int = 50,
                            with_enrichment: bool = False) -> str:
    """Submit a batch lead scoring task"""
    task = Task(
        task_id=f"batch_score_{uuid.uuid4().hex[:8]}",
        task_type="batch_lead_scoring",
        payload={
            'leads_data': leads_data,
            'batch_size': batch_size,
            'with_enrichment': with_enrichment
        },
        priority=priority,
        timeout=600  # 10 minutes for batch processing
    )
    
    return task_queue.enqueue_task(task)

def submit_enrichment_task(lead_data: Dict[str, Any], 
                         sources: List[str] = None,
                         priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """Submit a data enrichment task"""
    task = Task(
        task_id=f"enrich_{uuid.uuid4().hex[:8]}",
        task_type="data_enrichment",
        payload={
            'lead_data': lead_data,
            'sources': sources
        },
        priority=priority
    )
    
    return task_queue.enqueue_task(task)

# Example usage
async def example_usage():
    """Example of using the distributed processing system"""
    
    # Start worker manager
    manager_task = asyncio.create_task(worker_manager.start())
    
    try:
        # Submit some tasks
        lead_data = {
            'email': 'john.doe@example.com',
            'age': 35,
            'income': 75000
        }
        
        # Submit individual lead scoring
        task_id1 = submit_lead_scoring_task(lead_data, priority=TaskPriority.HIGH)
        print(f"Submitted lead scoring task: {task_id1}")
        
        # Submit batch processing
        batch_leads = [lead_data] * 100
        task_id2 = submit_batch_scoring_task(batch_leads, priority=TaskPriority.NORMAL)
        print(f"Submitted batch scoring task: {task_id2}")
        
        # Submit enrichment
        task_id3 = submit_enrichment_task(lead_data, sources=['clearbit', 'fullcontact'])
        print(f"Submitted enrichment task: {task_id3}")
        
        # Monitor for a while
        await asyncio.sleep(30)
        
        # Get stats
        queue_stats = task_queue.get_queue_stats()
        worker_stats = worker_manager.get_worker_stats()
        
        print("Queue Stats:", queue_stats)
        print("Worker Stats:", worker_stats)
        
    finally:
        # Stop worker manager
        await worker_manager.stop()
        manager_task.cancel()

if __name__ == "__main__":
    asyncio.run(example_usage())
