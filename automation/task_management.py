from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import logging

class TaskType(Enum):
    FOLLOW_UP_CALL = "follow_up_call"
    SEND_EMAIL = "send_email"
    SEND_QUOTE = "send_quote"
    SCHEDULE_MEETING = "schedule_meeting"
    RESEARCH_LEAD = "research_lead"
    UPDATE_CRM = "update_crm"
    CUSTOM = "custom"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"

@dataclass
class Task:
    task_id: str
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    assignee_id: str
    lead_id: Optional[str] = None
    due_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    completion_notes: str = ""
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskTemplate:
    template_id: str
    name: str
    task_type: TaskType
    title_template: str
    description_template: str
    default_priority: TaskPriority
    default_due_offset: Optional[timedelta] = None
    estimated_duration: Optional[timedelta] = None
    tags: List[str] = field(default_factory=list)

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.templates: Dict[str, TaskTemplate] = {}
        self.sla_rules: Dict[TaskType, timedelta] = {}
        
        # Initialize default templates
        self._create_default_templates()
        self._setup_default_slas()
    
    def _create_default_templates(self):
        """Create default task templates"""
        templates = [
            TaskTemplate(
                template_id="follow_up_call",
                name="Follow-up Call",
                task_type=TaskType.FOLLOW_UP_CALL,
                title_template="Follow-up call with {lead_name}",
                description_template="Call {lead_name} to discuss {product} options. Lead score: {score}",
                default_priority=TaskPriority.HIGH,
                default_due_offset=timedelta(hours=4),
                estimated_duration=timedelta(minutes=15),
                tags=["sales", "follow-up"]
            ),
            TaskTemplate(
                template_id="send_quote",
                name="Send Quote",
                task_type=TaskType.SEND_QUOTE,
                title_template="Send {product} quote to {lead_name}",
                description_template="Prepare and send personalized quote for {product}",
                default_priority=TaskPriority.HIGH,
                default_due_offset=timedelta(hours=2),
                estimated_duration=timedelta(minutes=30),
                tags=["sales", "quote"]
            ),
            TaskTemplate(
                template_id="research_lead",
                name="Research Lead",
                task_type=TaskType.RESEARCH_LEAD,
                title_template="Research background for {lead_name}",
                description_template="Research lead background and prepare talking points",
                default_priority=TaskPriority.MEDIUM,
                default_due_offset=timedelta(hours=1),
                estimated_duration=timedelta(minutes=20),
                tags=["research", "preparation"]
            )
        ]
        
        for template in templates:
            self.templates[template.template_id] = template
    
    def _setup_default_slas(self):
        """Setup default SLA rules"""
        self.sla_rules = {
            TaskType.FOLLOW_UP_CALL: timedelta(hours=4),
            TaskType.SEND_EMAIL: timedelta(hours=2),
            TaskType.SEND_QUOTE: timedelta(hours=6),
            TaskType.SCHEDULE_MEETING: timedelta(hours=24),
            TaskType.RESEARCH_LEAD: timedelta(hours=2),
            TaskType.UPDATE_CRM: timedelta(hours=1)
        }
    
    def create_task(self, title: str, description: str, task_type: TaskType,
                   assignee_id: str, priority: TaskPriority = TaskPriority.MEDIUM,
                   lead_id: Optional[str] = None, due_date: Optional[datetime] = None,
                   tags: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Create a new task"""
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Set due date based on SLA if not provided
        if not due_date and task_type in self.sla_rules:
            due_date = datetime.now() + self.sla_rules[task_type]
        
        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            assignee_id=assignee_id,
            lead_id=lead_id,
            due_date=due_date,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        logging.info(f"Created task: {task_id} - {title}")
        
        return task_id
    
    def create_task_from_template(self, template_id: str, assignee_id: str,
                                 lead_data: Dict[str, Any], 
                                 custom_params: Dict[str, Any] = None) -> str:
        """Create task from template"""
        
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        params = {
            'lead_name': lead_data.get('name', 'Unknown'),
            'product': lead_data.get('recommended_products', ['insurance'])[0],
            'score': lead_data.get('overall_score', 0),
            **(custom_params or {})
        }
        
        # Format title and description
        title = template.title_template.format(**params)
        description = template.description_template.format(**params)
        
        # Calculate due date
        due_date = None
        if template.default_due_offset:
            due_date = datetime.now() + template.default_due_offset
        
        return self.create_task(
            title=title,
            description=description,
            task_type=template.task_type,
            assignee_id=assignee_id,
            priority=template.default_priority,
            lead_id=lead_data.get('lead_id'),
            due_date=due_date,
            tags=template.tags.copy(),
            metadata={'template_id': template_id, 'lead_data': lead_data}
        )
    
    def update_task_status(self, task_id: str, status: TaskStatus,
                          completion_notes: str = "") -> bool:
        """Update task status"""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        old_status = task.status
        
        task.status = status
        task.updated_at = datetime.now()
        
        if status == TaskStatus.COMPLETED:
            task.completion_notes = completion_notes
            if task.created_at:
                task.actual_duration = datetime.now() - task.created_at
        
        logging.info(f"Updated task {task_id} status: {old_status.value} -> {status.value}")
        return True
    
    def assign_task(self, task_id: str, new_assignee_id: str) -> bool:
        """Reassign task to different user"""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        old_assignee = task.assignee_id
        
        task.assignee_id = new_assignee_id
        task.updated_at = datetime.now()
        
        logging.info(f"Reassigned task {task_id}: {old_assignee} -> {new_assignee_id}")
        return True
    
    def get_tasks_by_assignee(self, assignee_id: str, 
                             status_filter: List[TaskStatus] = None) -> List[Task]:
        """Get tasks assigned to specific user"""
        
        tasks = [task for task in self.tasks.values() 
                if task.assignee_id == assignee_id]
        
        if status_filter:
            tasks = [task for task in tasks if task.status in status_filter]
        
        # Sort by priority and due date
        tasks.sort(key=lambda x: (x.priority.value, x.due_date or datetime.max), reverse=True)
        
        return tasks
    
    def get_overdue_tasks(self) -> List[Task]:
        """Get all overdue tasks"""
        now = datetime.now()
        overdue_tasks = []
        
        for task in self.tasks.values():
            if (task.due_date and task.due_date < now and 
                task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]):
                
                # Update status to overdue
                task.status = TaskStatus.OVERDUE
                overdue_tasks.append(task)
        
        return overdue_tasks
    
    def get_task_analytics(self, assignee_id: Optional[str] = None,
                          date_range: Optional[tuple] = None) -> Dict[str, Any]:
        """Get task analytics"""
        
        tasks = list(self.tasks.values())
        
        # Filter by assignee
        if assignee_id:
            tasks = [task for task in tasks if task.assignee_id == assignee_id]
        
        # Filter by date range
        if date_range:
            start_date, end_date = date_range
            tasks = [task for task in tasks 
                    if start_date <= task.created_at <= end_date]
        
        if not tasks:
            return {"message": "No tasks found for the specified criteria"}
        
        # Calculate metrics
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
        overdue_tasks = len([t for t in tasks if t.status == TaskStatus.OVERDUE])
        
        # Task type distribution
        type_distribution = {}
        for task in tasks:
            task_type = task.task_type.value
            type_distribution[task_type] = type_distribution.get(task_type, 0) + 1
        
        # Priority distribution
        priority_distribution = {}
        for task in tasks:
            priority = task.priority.name
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        
        # Average completion time
        completed_with_duration = [t for t in tasks 
                                 if t.status == TaskStatus.COMPLETED and t.actual_duration]
        avg_completion_time = None
        if completed_with_duration:
            total_duration = sum([t.actual_duration.total_seconds() 
                                for t in completed_with_duration])
            avg_completion_time = total_duration / len(completed_with_duration) / 3600  # hours
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "overdue_tasks": overdue_tasks,
            "overdue_rate": overdue_tasks / total_tasks if total_tasks > 0 else 0,
            "type_distribution": type_distribution,
            "priority_distribution": priority_distribution,
            "average_completion_time_hours": avg_completion_time,
            "tasks_by_status": {
                status.value: len([t for t in tasks if t.status == status])
                for status in TaskStatus
            }
        }
    
    def create_recurring_tasks(self, template_id: str, assignee_id: str,
                              lead_ids: List[str], interval: timedelta) -> List[str]:
        """Create recurring tasks for multiple leads"""
        created_tasks = []
        
        for lead_id in lead_ids:
            # This would typically fetch lead data from database
            lead_data = {'lead_id': lead_id, 'name': f'Lead {lead_id}'}
            
            task_id = self.create_task_from_template(
                template_id, assignee_id, lead_data
            )
            created_tasks.append(task_id)
        
        logging.info(f"Created {len(created_tasks)} recurring tasks")
        return created_tasks
    
    def bulk_update_tasks(self, task_ids: List[str], updates: Dict[str, Any]) -> int:
        """Bulk update multiple tasks"""
        updated_count = 0
        
        for task_id in task_ids:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                for key, value in updates.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                
                task.updated_at = datetime.now()
                updated_count += 1
        
        logging.info(f"Bulk updated {updated_count} tasks")
        return updated_count