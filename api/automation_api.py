from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging

from automation.workflow_engine import (
    WorkflowEngine, TriggerType, ActionType, TriggerCondition, 
    WorkflowAction, RoutingStrategy, WorkflowStatus
)
from automation.lead_routing import LeadRouter, SalesRep, SalesRepStatus, RoutingRule
from automation.task_management import TaskManager, TaskType, TaskPriority, TaskStatus

router = APIRouter(prefix="/automation", tags=["Automation & Workflows"])

# Initialize automation components
workflow_engine = WorkflowEngine()
lead_router = LeadRouter()
task_manager = TaskManager()

# Pydantic Models
class WorkflowCreateRequest(BaseModel):
    name: str
    description: str
    trigger_type: TriggerType
    trigger_conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]

class TriggerWorkflowRequest(BaseModel):
    trigger_type: TriggerType
    trigger_data: Dict[str, Any]

class SalesRepRequest(BaseModel):
    name: str
    email: str
    specialties: List[str]
    territories: List[str]
    max_leads_per_day: int

class RouteLeadRequest(BaseModel):
    lead_data: Dict[str, Any]
    strategy: RoutingStrategy = RoutingStrategy.SCORE_BASED

class TaskCreateRequest(BaseModel):
    title: str
    description: str
    task_type: TaskType
    assignee_id: str
    priority: TaskPriority = TaskPriority.MEDIUM
    lead_id: Optional[str] = None
    due_date: Optional[datetime] = None
    tags: List[str] = []

class TaskUpdateRequest(BaseModel):
    status: TaskStatus
    completion_notes: str = ""

# Workflow Endpoints
@router.post("/workflows/create")
async def create_workflow(request: WorkflowCreateRequest):
    """Create a new workflow"""
    try:
        # Create trigger condition
        trigger = TriggerCondition(
            trigger_type=request.trigger_type,
            conditions=request.trigger_conditions
        )
        
        # Create actions
        actions = []
        for i, action_data in enumerate(request.actions):
            action = WorkflowAction(
                action_id=f"action_{i}",
                action_type=ActionType(action_data['action_type']),
                parameters=action_data.get('parameters', {}),
                conditions=action_data.get('conditions', {}),
                delay=timedelta(seconds=action_data.get('delay_seconds', 0)) if action_data.get('delay_seconds') else None
            )
            actions.append(action)
        
        # Create workflow
        workflow_id = workflow_engine.create_workflow(
            name=request.name,
            description=request.description,
            trigger=trigger,
            actions=actions
        )
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "message": f"Workflow '{request.name}' created successfully"
        }
        
    except Exception as e:
        logging.error(f"Error creating workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating workflow: {str(e)}")

@router.post("/workflows/{workflow_id}/activate")
async def activate_workflow(workflow_id: str):
    """Activate a workflow"""
    try:
        success = workflow_engine.activate_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "message": "Workflow activated successfully"
        }
        
    except Exception as e:
        logging.error(f"Error activating workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error activating workflow: {str(e)}")

@router.post("/workflows/trigger")
async def trigger_workflows(request: TriggerWorkflowRequest):
    """Trigger workflows based on event"""
    try:
        execution_ids = await workflow_engine.trigger_workflow(
            trigger_type=request.trigger_type,
            trigger_data=request.trigger_data
        )
        
        return {
            "status": "success",
            "triggered_executions": execution_ids,
            "count": len(execution_ids),
            "message": f"Triggered {len(execution_ids)} workflow executions"
        }
        
    except Exception as e:
        logging.error(f"Error triggering workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error triggering workflows: {str(e)}")

@router.get("/workflows/{workflow_id}/stats")
async def get_workflow_stats(workflow_id: str):
    """Get workflow execution statistics"""
    try:
        stats = workflow_engine.get_workflow_stats(workflow_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "status": "success",
            "stats": stats
        }
        
    except Exception as e:
        logging.error(f"Error getting workflow stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting workflow stats: {str(e)}")

# Lead Routing Endpoints
@router.post("/routing/sales-reps")
async def add_sales_rep(request: SalesRepRequest):
    """Add a new sales representative"""
    try:
        rep = SalesRep(
            rep_id=f"rep_{request.email.split('@')[0]}",
            name=request.name,
            email=request.email,
            status=SalesRepStatus.AVAILABLE,
            specialties=request.specialties,
            territories=request.territories,
            max_leads_per_day=request.max_leads_per_day
        )
        
        lead_router.add_sales_rep(rep)
        
        return {
            "status": "success",
            "rep_id": rep.rep_id,
            "message": f"Sales rep '{request.name}' added successfully"
        }
        
    except Exception as e:
        logging.error(f"Error adding sales rep: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding sales rep: {str(e)}")

@router.post("/routing/route-lead")
async def route_lead(request: RouteLeadRequest):
    """Route a lead to the best available sales rep"""
    try:
        assigned_rep_id = lead_router.route_lead(
            lead_data=request.lead_data,
            strategy=request.strategy
        )
        
        if not assigned_rep_id:
            return {
                "status": "warning",
                "assigned_rep_id": None,
                "message": "No available sales reps for routing"
            }
        
        # Create follow-up task automatically
        task_id = task_manager.create_task_from_template(
            template_id="follow_up_call",
            assignee_id=assigned_rep_id,
            lead_data=request.lead_data
        )
        
        return {
            "status": "success",
            "assigned_rep_id": assigned_rep_id,
            "created_task_id": task_id,
            "strategy_used": request.strategy.value,
            "message": f"Lead routed to rep {assigned_rep_id} with follow-up task created"
        }
        
    except Exception as e:
        logging.error(f"Error routing lead: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error routing lead: {str(e)}")

@router.put("/routing/sales-reps/{rep_id}/status")
async def update_rep_status(rep_id: str, status: SalesRepStatus):
    """Update sales rep availability status"""
    try:
        lead_router.update_rep_status(rep_id, status)
        
        return {
            "status": "success",
            "rep_id": rep_id,
            "new_status": status.value,
            "message": f"Rep status updated to {status.value}"
        }
        
    except Exception as e:
        logging.error(f"Error updating rep status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating rep status: {str(e)}")

@router.get("/routing/analytics")
async def get_routing_analytics():
    """Get lead routing analytics"""
    try:
        analytics = lead_router.get_routing_analytics()
        
        return {
            "status": "success",
            "analytics": analytics
        }
        
    except Exception as e:
        logging.error(f"Error getting routing analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting routing analytics: {str(e)}")

# Task Management Endpoints
@router.post("/tasks/create")
async def create_task(request: TaskCreateRequest):
    """Create a new task"""
    try:
        task_id = task_manager.create_task(
            title=request.title,
            description=request.description,
            task_type=request.task_type,
            assignee_id=request.assignee_id,
            priority=request.priority,
            lead_id=request.lead_id,
            due_date=request.due_date,
            tags=request.tags
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Task created successfully"
        }
        
    except Exception as e:
        logging.error(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")

@router.post("/tasks/from-template/{template_id}")
async def create_task_from_template(template_id: str, assignee_id: str, lead_data: Dict[str, Any]):
    """Create task from template"""
    try:
        task_id = task_manager.create_task_from_template(
            template_id=template_id,
            assignee_id=assignee_id,
            lead_data=lead_data
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "template_used": template_id,
            "message": "Task created from template successfully"
        }
        
    except Exception as e:
        logging.error(f"Error creating task from template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating task from template: {str(e)}")

@router.put("/tasks/{task_id}/status")
async def update_task_status(task_id: str, request: TaskUpdateRequest):
    """Update task status"""
    try:
        success = task_manager.update_task_status(
            task_id=task_id,
            status=request.status,
            completion_notes=request.completion_notes
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "status": "success",
            "task_id": task_id,
            "new_status": request.status.value,
            "message": "Task status updated successfully"
        }
        
    except Exception as e:
        logging.error(f"Error updating task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating task status: {str(e)}")

@router.get("/tasks/assignee/{assignee_id}")
async def get_tasks_by_assignee(assignee_id: str, status_filter: Optional[List[TaskStatus]] = None):
    """Get tasks assigned to specific user"""
    try:
        tasks = task_manager.get_tasks_by_assignee(assignee_id, status_filter)
        
        # Convert to dict for JSON serialization
        tasks_data = []
        for task in tasks:
            task_dict = {
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "task_type": task.task_type.value,
                "priority": task.priority.value,
                "status": task.status.value,
                "due_date": task.due_date.isoformat() if task.due_date else None,
                "created_at": task.created_at.isoformat(),
                "lead_id": task.lead_id,
                "tags": task.tags
            }
            tasks_data.append(task_dict)
        
        return {
            "status": "success",
            "assignee_id": assignee_id,
            "tasks": tasks_data,
            "count": len(tasks_data)
        }
        
    except Exception as e:
        logging.error(f"Error getting tasks by assignee: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting tasks by assignee: {str(e)}")

@router.get("/tasks/overdue")
async def get_overdue_tasks():
    """Get all overdue tasks"""
    try:
        overdue_tasks = task_manager.get_overdue_tasks()
        
        # Convert to dict for JSON serialization
        tasks_data = []
        for task in overdue_tasks:
            task_dict = {
                "task_id": task.task_id,
                "title": task.title,
                "assignee_id": task.assignee_id,
                "due_date": task.due_date.isoformat() if task.due_date else None,
                "priority": task.priority.value,
                "lead_id": task.lead_id,
                "days_overdue": (datetime.now() - task.due_date).days if task.due_date else 0
            }
            tasks_data.append(task_dict)
        
        return {
            "status": "success",
            "overdue_tasks": tasks_data,
            "count": len(tasks_data)
        }
        
    except Exception as e:
        logging.error(f"Error getting overdue tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting overdue tasks: {str(e)}")

@router.get("/tasks/analytics")
async def get_task_analytics(assignee_id: Optional[str] = None):
    """Get task analytics"""
    try:
        analytics = task_manager.get_task_analytics(assignee_id=assignee_id)
        
        return {
            "status": "success",
            "analytics": analytics
        }
        
    except Exception as e:
        logging.error(f"Error getting task analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting task analytics: {str(e)}")

# Integrated Automation Endpoints
@router.post("/lead-to-workflow")
async def process_lead_through_automation(lead_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Process a lead through the complete automation pipeline"""
    try:
        results = {}
        
        # 1. Route the lead
        assigned_rep_id = lead_router.route_lead(lead_data)
        results['routing'] = {
            'assigned_rep_id': assigned_rep_id,
            'strategy': 'score_based'
        }
        
        # 2. Trigger workflows
        execution_ids = await workflow_engine.trigger_workflow(
            trigger_type=TriggerType.LEAD_SCORED,
            trigger_data=lead_data
        )
        results['workflows'] = {
            'triggered_executions': execution_ids,
            'count': len(execution_ids)
        }
        
        # 3. Create initial tasks
        if assigned_rep_id:
            task_id = task_manager.create_task_from_template(
                template_id="follow_up_call",
                assignee_id=assigned_rep_id,
                lead_data=lead_data
            )
            results['tasks'] = {
                'created_task_id': task_id
            }
        
        # 4. Schedule background analytics update
        background_tasks.add_task(_update_automation_analytics, lead_data, results)
        
        return {
            "status": "success",
            "lead_id": lead_data.get('lead_id'),
            "automation_results": results,
            "message": "Lead processed through automation pipeline successfully"
        }
        
    except Exception as e:
        logging.error(f"Error processing lead through automation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing lead: {str(e)}")

# Background task functions
async def _update_automation_analytics(lead_data: Dict[str, Any], results: Dict[str, Any]):
    """Update automation analytics in background"""
    try:
        # This would typically update analytics database
        logging.info(f"Updated automation analytics for lead {lead_data.get('lead_id')}")
    except Exception as e:
        logging.error(f"Error updating automation analytics: {str(e)}")