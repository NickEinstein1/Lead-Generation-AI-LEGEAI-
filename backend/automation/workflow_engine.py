from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
import json
import uuid

class TriggerType(Enum):
    LEAD_SCORED = "lead_scored"
    EMAIL_OPENED = "email_opened"
    EMAIL_CLICKED = "email_clicked"
    FORM_SUBMITTED = "form_submitted"
    TIME_BASED = "time_based"
    SCORE_THRESHOLD = "score_threshold"
    LEAD_STATUS_CHANGE = "lead_status_change"
    MANUAL_TRIGGER = "manual_trigger"

class ActionType(Enum):
    SEND_EMAIL = "send_email"
    ASSIGN_LEAD = "assign_lead"
    UPDATE_LEAD_STATUS = "update_lead_status"
    CREATE_TASK = "create_task"
    SEND_NOTIFICATION = "send_notification"
    SCHEDULE_CALL = "schedule_call"
    ADD_TO_SEQUENCE = "add_to_sequence"
    REMOVE_FROM_SEQUENCE = "remove_from_sequence"
    UPDATE_SCORE = "update_score"
    WEBHOOK_CALL = "webhook_call"

class WorkflowStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TriggerCondition:
    trigger_type: TriggerType
    conditions: Dict[str, Any]
    delay: Optional[timedelta] = None

@dataclass
class WorkflowAction:
    action_id: str
    action_type: ActionType
    parameters: Dict[str, Any]
    conditions: Dict[str, Any] = field(default_factory=dict)
    delay: Optional[timedelta] = None

@dataclass
class Workflow:
    workflow_id: str
    name: str
    description: str
    trigger: TriggerCondition
    actions: List[WorkflowAction]
    status: WorkflowStatus = WorkflowStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0

@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    lead_id: str
    trigger_data: Dict[str, Any]
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_action_index: int = 0
    execution_log: List[Dict[str, Any]] = field(default_factory=list)

class WorkflowEngine:
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.action_handlers: Dict[ActionType, Callable] = {}
        self.running = False
        
        # Register default action handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default action handlers"""
        self.action_handlers[ActionType.SEND_EMAIL] = self._handle_send_email
        self.action_handlers[ActionType.ASSIGN_LEAD] = self._handle_assign_lead
        self.action_handlers[ActionType.UPDATE_LEAD_STATUS] = self._handle_update_status
        self.action_handlers[ActionType.CREATE_TASK] = self._handle_create_task
        self.action_handlers[ActionType.SEND_NOTIFICATION] = self._handle_send_notification
        self.action_handlers[ActionType.SCHEDULE_CALL] = self._handle_schedule_call
        self.action_handlers[ActionType.WEBHOOK_CALL] = self._handle_webhook_call
    
    def create_workflow(self, name: str, description: str, 
                       trigger: TriggerCondition, 
                       actions: List[WorkflowAction]) -> str:
        """Create a new workflow"""
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            trigger=trigger,
            actions=actions
        )
        
        self.workflows[workflow_id] = workflow
        logging.info(f"Created workflow: {workflow_id} - {name}")
        
        return workflow_id
    
    def activate_workflow(self, workflow_id: str) -> bool:
        """Activate a workflow"""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].status = WorkflowStatus.ACTIVE
            self.workflows[workflow_id].updated_at = datetime.now()
            logging.info(f"Activated workflow: {workflow_id}")
            return True
        return False
    
    async def trigger_workflow(self, trigger_type: TriggerType, 
                              trigger_data: Dict[str, Any]) -> List[str]:
        """Trigger workflows based on event"""
        triggered_executions = []
        
        for workflow in self.workflows.values():
            if (workflow.status == WorkflowStatus.ACTIVE and 
                workflow.trigger.trigger_type == trigger_type):
                
                # Check if trigger conditions are met
                if self._evaluate_trigger_conditions(workflow.trigger, trigger_data):
                    execution_id = await self._start_workflow_execution(workflow, trigger_data)
                    triggered_executions.append(execution_id)
        
        return triggered_executions
    
    def _evaluate_trigger_conditions(self, trigger: TriggerCondition, 
                                   trigger_data: Dict[str, Any]) -> bool:
        """Evaluate if trigger conditions are met"""
        conditions = trigger.conditions
        
        for key, expected_value in conditions.items():
            if key not in trigger_data:
                return False
            
            actual_value = trigger_data[key]
            
            # Handle different condition types
            if isinstance(expected_value, dict):
                operator = expected_value.get('operator', 'equals')
                value = expected_value.get('value')
                
                if operator == 'equals' and actual_value != value:
                    return False
                elif operator == 'greater_than' and actual_value <= value:
                    return False
                elif operator == 'less_than' and actual_value >= value:
                    return False
                elif operator == 'contains' and value not in str(actual_value):
                    return False
            else:
                if actual_value != expected_value:
                    return False
        
        return True
    
    async def _start_workflow_execution(self, workflow: Workflow, 
                                       trigger_data: Dict[str, Any]) -> str:
        """Start executing a workflow"""
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        lead_id = trigger_data.get('lead_id', 'unknown')
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow.workflow_id,
            lead_id=lead_id,
            trigger_data=trigger_data,
            status=WorkflowStatus.ACTIVE,
            started_at=datetime.now()
        )
        
        self.executions[execution_id] = execution
        workflow.execution_count += 1
        
        # Start executing actions
        asyncio.create_task(self._execute_workflow_actions(execution))
        
        logging.info(f"Started workflow execution: {execution_id}")
        return execution_id
    
    async def _execute_workflow_actions(self, execution: WorkflowExecution):
        """Execute workflow actions sequentially"""
        workflow = self.workflows[execution.workflow_id]
        
        try:
            for i, action in enumerate(workflow.actions):
                execution.current_action_index = i
                
                # Apply delay if specified
                if action.delay:
                    await asyncio.sleep(action.delay.total_seconds())
                
                # Check action conditions
                if not self._evaluate_action_conditions(action, execution):
                    self._log_execution(execution, f"Skipped action {action.action_id} - conditions not met")
                    continue
                
                # Execute action
                success = await self._execute_action(action, execution)
                
                if not success:
                    execution.status = WorkflowStatus.FAILED
                    workflow.failure_count += 1
                    self._log_execution(execution, f"Action {action.action_id} failed")
                    break
                
                self._log_execution(execution, f"Action {action.action_id} completed successfully")
            
            # Mark as completed if all actions succeeded
            if execution.status == WorkflowStatus.ACTIVE:
                execution.status = WorkflowStatus.COMPLETED
                workflow.success_count += 1
            
            execution.completed_at = datetime.now()
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            workflow.failure_count += 1
            self._log_execution(execution, f"Workflow execution failed: {str(e)}")
            logging.error(f"Workflow execution {execution.execution_id} failed: {str(e)}")
    
    def _evaluate_action_conditions(self, action: WorkflowAction, 
                                   execution: WorkflowExecution) -> bool:
        """Evaluate if action conditions are met"""
        if not action.conditions:
            return True
        
        # Get current lead data (would typically fetch from database)
        lead_data = execution.trigger_data
        
        return self._evaluate_trigger_conditions(
            TriggerCondition(TriggerType.MANUAL_TRIGGER, action.conditions),
            lead_data
        )
    
    async def _execute_action(self, action: WorkflowAction, 
                             execution: WorkflowExecution) -> bool:
        """Execute a single action"""
        handler = self.action_handlers.get(action.action_type)
        
        if not handler:
            logging.error(f"No handler found for action type: {action.action_type}")
            return False
        
        try:
            result = await handler(action, execution)
            return result
        except Exception as e:
            logging.error(f"Action execution failed: {str(e)}")
            return False
    
    def _log_execution(self, execution: WorkflowExecution, message: str):
        """Log execution step"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "action_index": execution.current_action_index
        }
        execution.execution_log.append(log_entry)
    
    # Action Handlers
    async def _handle_send_email(self, action: WorkflowAction, 
                                execution: WorkflowExecution) -> bool:
        """Handle send email action"""
        # Integration with email system
        logging.info(f"Sending email to lead {execution.lead_id}")
        return True
    
    async def _handle_assign_lead(self, action: WorkflowAction, 
                                 execution: WorkflowExecution) -> bool:
        """Handle lead assignment action"""
        assignee = action.parameters.get('assignee')
        logging.info(f"Assigning lead {execution.lead_id} to {assignee}")
        return True
    
    async def _handle_update_status(self, action: WorkflowAction, 
                                   execution: WorkflowExecution) -> bool:
        """Handle lead status update action"""
        new_status = action.parameters.get('status')
        logging.info(f"Updating lead {execution.lead_id} status to {new_status}")
        return True
    
    async def _handle_create_task(self, action: WorkflowAction, 
                                 execution: WorkflowExecution) -> bool:
        """Handle task creation action"""
        task_type = action.parameters.get('task_type')
        assignee = action.parameters.get('assignee')
        logging.info(f"Creating {task_type} task for {assignee} - Lead {execution.lead_id}")
        return True
    
    async def _handle_send_notification(self, action: WorkflowAction, 
                                       execution: WorkflowExecution) -> bool:
        """Handle notification sending action"""
        recipient = action.parameters.get('recipient')
        message = action.parameters.get('message')
        logging.info(f"Sending notification to {recipient}: {message}")
        return True
    
    async def _handle_schedule_call(self, action: WorkflowAction, 
                                   execution: WorkflowExecution) -> bool:
        """Handle call scheduling action"""
        assignee = action.parameters.get('assignee')
        scheduled_time = action.parameters.get('scheduled_time')
        logging.info(f"Scheduling call for {assignee} with lead {execution.lead_id} at {scheduled_time}")
        return True
    
    async def _handle_webhook_call(self, action: WorkflowAction, 
                                  execution: WorkflowExecution) -> bool:
        """Handle webhook call action"""
        url = action.parameters.get('url')
        logging.info(f"Calling webhook: {url}")
        return True
    
    def get_workflow_stats(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        if workflow_id not in self.workflows:
            return {}
        
        workflow = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "execution_count": workflow.execution_count,
            "success_count": workflow.success_count,
            "failure_count": workflow.failure_count,
            "success_rate": workflow.success_count / workflow.execution_count if workflow.execution_count > 0 else 0,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat()
        }