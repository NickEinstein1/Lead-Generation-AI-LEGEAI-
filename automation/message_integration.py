from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import logging

from automation.workflow_engine import WorkflowEngine, TriggerType, ActionType, WorkflowAction, TriggerCondition
from automation.notification_system import NotificationSystem, NotificationType, NotificationPriority
from automation.task_management import TaskManager, TaskType, TaskPriority

class MessageAutomationIntegration:
    """Integration between automation system and message generation"""
    
    def __init__(self, workflow_engine: WorkflowEngine, 
                 notification_system: NotificationSystem,
                 task_manager: TaskManager):
        self.workflow_engine = workflow_engine
        self.notification_system = notification_system
        self.task_manager = task_manager
        
        # Setup message-specific workflows
        self._setup_message_workflows()
        
        # Setup message templates
        self._setup_message_templates()
    
    def _setup_message_workflows(self):
        """Setup automated workflows for message generation"""
        
        # High-score lead workflow
        high_score_trigger = TriggerCondition(
            trigger_type=TriggerType.LEAD_SCORED,
            conditions={
                "overall_score": {"operator": "greater_than", "value": 80},
                "priority_level": "HIGH"
            }
        )
        
        high_score_actions = [
            WorkflowAction(
                action_id="generate_urgent_message",
                action_type=ActionType.SEND_EMAIL,
                parameters={
                    "message_type": "urgent_follow_up",
                    "template": "high_score_lead",
                    "priority": "urgent"
                }
            ),
            WorkflowAction(
                action_id="create_immediate_task",
                action_type=ActionType.CREATE_TASK,
                parameters={
                    "task_type": "follow_up_call",
                    "priority": "urgent",
                    "due_hours": 1
                }
            ),
            WorkflowAction(
                action_id="notify_manager",
                action_type=ActionType.SEND_NOTIFICATION,
                parameters={
                    "recipient": "sales_manager@company.com",
                    "type": "sms",
                    "message": "High-score lead requires immediate attention"
                }
            )
        ]
        
        self.workflow_engine.create_workflow(
            name="High Score Lead Response",
            description="Automated response for high-scoring leads",
            trigger=high_score_trigger,
            actions=high_score_actions
        )
        
        # Email engagement workflow
        email_engagement_trigger = TriggerCondition(
            trigger_type=TriggerType.EMAIL_OPENED,
            conditions={}
        )
        
        email_engagement_actions = [
            WorkflowAction(
                action_id="generate_follow_up",
                action_type=ActionType.SEND_EMAIL,
                parameters={
                    "message_type": "engagement_follow_up",
                    "delay_hours": 2
                },
                delay=timedelta(hours=2)
            ),
            WorkflowAction(
                action_id="update_engagement_score",
                action_type=ActionType.UPDATE_SCORE,
                parameters={
                    "score_boost": 5,
                    "reason": "email_engagement"
                }
            )
        ]
        
        self.workflow_engine.create_workflow(
            name="Email Engagement Follow-up",
            description="Follow-up sequence for email engagement",
            trigger=email_engagement_trigger,
            actions=email_engagement_actions
        )
        
        # Nurture sequence workflow
        nurture_trigger = TriggerCondition(
            trigger_type=TriggerType.LEAD_SCORED,
            conditions={
                "overall_score": {"operator": "less_than", "value": 60},
                "priority_level": "MEDIUM"
            }
        )
        
        nurture_actions = [
            WorkflowAction(
                action_id="start_nurture_sequence",
                action_type=ActionType.ADD_TO_SEQUENCE,
                parameters={
                    "sequence_type": "educational_nurture",
                    "duration_days": 14
                }
            ),
            WorkflowAction(
                action_id="schedule_check_in",
                action_type=ActionType.CREATE_TASK,
                parameters={
                    "task_type": "research_lead",
                    "due_days": 7
                },
                delay=timedelta(days=7)
            )
        ]
        
        self.workflow_engine.create_workflow(
            name="Lead Nurture Sequence",
            description="Automated nurture sequence for medium-score leads",
            trigger=nurture_trigger,
            actions=nurture_actions
        )
    
    def _setup_message_templates(self):
        """Setup message-specific notification templates"""
        from automation.notification_system import NotificationTemplate
        
        templates = [
            NotificationTemplate(
                template_id="message_generation_success",
                name="Message Generated Successfully",
                notification_type=NotificationType.IN_APP,
                subject_template="Message Generated for {lead_name}",
                body_template="Personalized {message_type} message generated for {lead_name}. Message ID: {message_id}",
                priority=NotificationPriority.LOW
            ),
            NotificationTemplate(
                template_id="sequence_completed",
                name="Message Sequence Completed",
                notification_type=NotificationType.EMAIL,
                subject_template="Message Sequence Completed: {lead_name}",
                body_template="The {sequence_type} sequence for {lead_name} has been completed. Total messages sent: {message_count}. Next action: {next_action}",
                priority=NotificationPriority.MEDIUM
            ),
            NotificationTemplate(
                template_id="ab_test_results",
                name="A/B Test Results Available",
                notification_type=NotificationType.EMAIL,
                subject_template="A/B Test Results: {test_name}",
                body_template="A/B test '{test_name}' has reached statistical significance. Winner: Variant {winning_variant} with {improvement}% improvement.",
                priority=NotificationPriority.HIGH
            )
        ]
        
        for template in templates:
            self.notification_system.add_template(template)
    
    async def process_message_generation_event(self, event_type: str, event_data: Dict[str, Any]):
        """Process message generation events and trigger appropriate workflows"""
        
        try:
            if event_type == "message_generated":
                await self._handle_message_generated(event_data)
            elif event_type == "sequence_started":
                await self._handle_sequence_started(event_data)
            elif event_type == "sequence_completed":
                await self._handle_sequence_completed(event_data)
            elif event_type == "ab_test_completed":
                await self._handle_ab_test_completed(event_data)
            elif event_type == "email_engagement":
                await self._handle_email_engagement(event_data)
            
        except Exception as e:
            logging.error(f"Error processing message generation event {event_type}: {str(e)}")
    
    async def _handle_message_generated(self, event_data: Dict[str, Any]):
        """Handle message generation completion"""
        
        # Create follow-up task if high priority
        if event_data.get('priority_level') in ['HIGH', 'CRITICAL']:
            task_id = self.task_manager.create_task_from_template(
                template_id="follow_up_call",
                assignee_id=event_data.get('assigned_rep_id'),
                lead_data=event_data
            )
            
            # Notify assignee
            await self.notification_system.send_from_template(
                template_id="message_generation_success",
                recipient=event_data.get('assignee_email', 'unknown@company.com'),
                template_data=event_data
            )
        
        # Trigger workflow for message follow-up
        await self.workflow_engine.trigger_workflow(
            trigger_type=TriggerType.MANUAL_TRIGGER,
            trigger_data={
                **event_data,
                "event_type": "message_generated"
            }
        )
    
    async def _handle_sequence_started(self, event_data: Dict[str, Any]):
        """Handle message sequence start"""
        
        # Create monitoring task
        task_id = self.task_manager.create_task(
            title=f"Monitor sequence for {event_data.get('lead_name', 'Unknown')}",
            description=f"Monitor {event_data.get('sequence_type')} sequence progress",
            task_type=TaskType.UPDATE_CRM,
            assignee_id=event_data.get('assigned_rep_id'),
            priority=TaskPriority.LOW,
            lead_id=event_data.get('lead_id'),
            due_date=datetime.now() + timedelta(days=7)
        )
        
        logging.info(f"Created sequence monitoring task: {task_id}")
    
    async def _handle_sequence_completed(self, event_data: Dict[str, Any]):
        """Handle message sequence completion"""
        
        # Notify assignee of completion
        await self.notification_system.send_from_template(
            template_id="sequence_completed",
            recipient=event_data.get('assignee_email', 'unknown@company.com'),
            template_data=event_data
        )
        
        # Create follow-up task based on sequence results
        next_action = self._determine_next_action(event_data)
        
        if next_action:
            task_id = self.task_manager.create_task(
                title=f"Follow-up: {next_action['title']}",
                description=next_action['description'],
                task_type=TaskType(next_action['task_type']),
                assignee_id=event_data.get('assigned_rep_id'),
                priority=TaskPriority(next_action['priority']),
                lead_id=event_data.get('lead_id'),
                due_date=datetime.now() + timedelta(days=next_action['due_days'])
            )
            
            logging.info(f"Created follow-up task after sequence completion: {task_id}")
    
    async def _handle_ab_test_completed(self, event_data: Dict[str, Any]):
        """Handle A/B test completion"""
        
        # Notify marketing team of results
        await self.notification_system.send_from_template(
            template_id="ab_test_results",
            recipient="marketing@company.com",
            template_data=event_data
        )
        
        # Create task to implement winning variant
        if event_data.get('winning_variant'):
            task_id = self.task_manager.create_task(
                title=f"Implement A/B test winner: {event_data.get('test_name')}",
                description=f"Implement variant {event_data.get('winning_variant')} as the new default",
                task_type=TaskType.UPDATE_CRM,
                assignee_id="marketing_manager",
                priority=TaskPriority.MEDIUM,
                due_date=datetime.now() + timedelta(days=3)
            )
    
    async def _handle_email_engagement(self, event_data: Dict[str, Any]):
        """Handle email engagement events"""
        
        # Trigger engagement workflow
        await self.workflow_engine.trigger_workflow(
            trigger_type=TriggerType.EMAIL_OPENED,
            trigger_data=event_data
        )
        
        # Update lead score based on engagement
        engagement_boost = self._calculate_engagement_boost(event_data)
        
        if engagement_boost > 0:
            # This would integrate with the scoring system
            logging.info(f"Boosting lead score by {engagement_boost} for engagement")
    
    def _determine_next_action(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine next action based on sequence results"""
        
        sequence_type = event_data.get('sequence_type')
        engagement_rate = event_data.get('engagement_rate', 0)
        conversion_probability = event_data.get('conversion_probability', 0)
        
        if sequence_type == "educational_nurture":
            if engagement_rate > 0.3:
                return {
                    'title': 'Schedule sales call',
                    'description': 'Lead showed high engagement, schedule a sales call',
                    'task_type': 'follow_up_call',
                    'priority': 'high',
                    'due_days': 2
                }
            elif engagement_rate > 0.1:
                return {
                    'title': 'Send product demo',
                    'description': 'Lead showed moderate engagement, send product demo',
                    'task_type': 'send_email',
                    'priority': 'medium',
                    'due_days': 5
                }
        
        elif sequence_type == "urgent_follow_up":
            if conversion_probability > 0.7:
                return {
                    'title': 'Immediate sales call',
                    'description': 'High conversion probability, call immediately',
                    'task_type': 'follow_up_call',
                    'priority': 'urgent',
                    'due_days': 1
                }
        
        return None
    
    def _calculate_engagement_boost(self, event_data: Dict[str, Any]) -> int:
        """Calculate score boost based on engagement type"""
        
        engagement_type = event_data.get('engagement_type')
        
        engagement_scores = {
            'email_open': 2,
            'email_click': 5,
            'website_visit': 3,
            'document_download': 7,
            'video_watch': 4,
            'form_submission': 10
        }
        
        return engagement_scores.get(engagement_type, 0)

# Global integration instance
message_integration = None

def initialize_message_integration(workflow_engine: WorkflowEngine,
                                 notification_system: NotificationSystem,
                                 task_manager: TaskManager):
    """Initialize the global message integration"""
    global message_integration
    message_integration = MessageAutomationIntegration(
        workflow_engine, notification_system, task_manager
    )
    return message_integration
