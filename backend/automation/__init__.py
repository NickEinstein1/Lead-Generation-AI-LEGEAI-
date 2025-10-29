"""
Automation System for Insurance Lead Scoring Platform

This module provides comprehensive automation capabilities including:
- Workflow engine for automated processes
- Lead routing and assignment
- Task management and tracking
- Notification system
- Message generation integration
- Dashboard and analytics
"""

from .workflow_engine import (
    WorkflowEngine, TriggerType, ActionType, WorkflowStatus,
    TriggerCondition, WorkflowAction, WorkflowExecution
)
from .lead_routing import (
    LeadRouter, SalesRep, SalesRepStatus, RoutingStrategy, RoutingRule
)
from .task_management import (
    TaskManager, Task, TaskType, TaskPriority, TaskStatus
)
from .notification_system import (
    NotificationSystem, NotificationType, NotificationPriority,
    NotificationTemplate, Notification
)
from .message_integration import (
    MessageAutomationIntegration, initialize_message_integration
)

# Global instances
workflow_engine = WorkflowEngine()
lead_router = LeadRouter()
task_manager = TaskManager()
notification_system = NotificationSystem()

# Initialize message integration
message_integration = initialize_message_integration(
    workflow_engine, notification_system, task_manager
)

__all__ = [
    # Core engines
    'workflow_engine',
    'lead_router', 
    'task_manager',
    'notification_system',
    'message_integration',
    
    # Classes
    'WorkflowEngine',
    'LeadRouter',
    'TaskManager', 
    'NotificationSystem',
    'MessageAutomationIntegration',
    
    # Enums
    'TriggerType',
    'ActionType', 
    'WorkflowStatus',
    'SalesRepStatus',
    'RoutingStrategy',
    'TaskType',
    'TaskPriority', 
    'TaskStatus',
    'NotificationType',
    'NotificationPriority',
    
    # Data classes
    'TriggerCondition',
    'WorkflowAction',
    'WorkflowExecution', 
    'SalesRep',
    'RoutingRule',
    'Task',
    'NotificationTemplate',
    'Notification'
]

# Version info
__version__ = "1.0.0"
__author__ = "Insurance Lead Scoring Platform"