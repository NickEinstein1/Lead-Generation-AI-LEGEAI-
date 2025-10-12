from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
import json
import uuid
from abc import ABC, abstractmethod

class NotificationType(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    SLACK = "slack"
    WEBHOOK = "webhook"
    IN_APP = "in_app"

class NotificationPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class NotificationStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class NotificationTemplate:
    template_id: str
    name: str
    notification_type: NotificationType
    subject_template: str
    body_template: str
    priority: NotificationPriority = NotificationPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Notification:
    notification_id: str
    notification_type: NotificationType
    recipient: str
    subject: str
    body: str
    priority: NotificationPriority
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

class NotificationChannel(ABC):
    """Abstract base class for notification channels"""
    
    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send notification through this channel"""
        pass
    
    @abstractmethod
    def validate_recipient(self, recipient: str) -> bool:
        """Validate recipient format for this channel"""
        pass

class EmailChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
    
    async def send(self, notification: Notification) -> bool:
        """Send email notification"""
        try:
            # Simulate email sending
            await asyncio.sleep(0.1)  # Simulate network delay
            logging.info(f"Email sent to {notification.recipient}: {notification.subject}")
            return True
        except Exception as e:
            logging.error(f"Failed to send email: {str(e)}")
            return False
    
    def validate_recipient(self, recipient: str) -> bool:
        """Validate email address"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, recipient))

class SMSChannel(NotificationChannel):
    """SMS notification channel"""
    
    def __init__(self, sms_config: Dict[str, Any]):
        self.sms_config = sms_config
    
    async def send(self, notification: Notification) -> bool:
        """Send SMS notification"""
        try:
            # Simulate SMS sending
            await asyncio.sleep(0.1)
            logging.info(f"SMS sent to {notification.recipient}: {notification.body[:50]}...")
            return True
        except Exception as e:
            logging.error(f"Failed to send SMS: {str(e)}")
            return False
    
    def validate_recipient(self, recipient: str) -> bool:
        """Validate phone number"""
        import re
        pattern = r'^\+?1?[2-9]\d{2}[2-9]\d{2}\d{4}$'
        return bool(re.match(pattern, recipient.replace('-', '').replace(' ', '')))

class SlackChannel(NotificationChannel):
    """Slack notification channel"""
    
    def __init__(self, slack_config: Dict[str, Any]):
        self.slack_config = slack_config
    
    async def send(self, notification: Notification) -> bool:
        """Send Slack notification"""
        try:
            # Simulate Slack API call
            await asyncio.sleep(0.1)
            logging.info(f"Slack message sent to {notification.recipient}")
            return True
        except Exception as e:
            logging.error(f"Failed to send Slack message: {str(e)}")
            return False
    
    def validate_recipient(self, recipient: str) -> bool:
        """Validate Slack channel or user"""
        return recipient.startswith('#') or recipient.startswith('@')

class WebhookChannel(NotificationChannel):
    """Webhook notification channel"""
    
    def __init__(self, webhook_config: Dict[str, Any]):
        self.webhook_config = webhook_config
    
    async def send(self, notification: Notification) -> bool:
        """Send webhook notification"""
        try:
            # Simulate HTTP request
            await asyncio.sleep(0.1)
            logging.info(f"Webhook called: {notification.recipient}")
            return True
        except Exception as e:
            logging.error(f"Failed to call webhook: {str(e)}")
            return False
    
    def validate_recipient(self, recipient: str) -> bool:
        """Validate webhook URL"""
        return recipient.startswith('http://') or recipient.startswith('https://')

class NotificationSystem:
    """Central notification management system"""
    
    def __init__(self):
        self.channels: Dict[NotificationType, NotificationChannel] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.notifications: Dict[str, Notification] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
        
        # Initialize default templates
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default notification templates"""
        templates = [
            NotificationTemplate(
                template_id="lead_assigned",
                name="Lead Assigned",
                notification_type=NotificationType.EMAIL,
                subject_template="New Lead Assigned: {lead_name}",
                body_template="You have been assigned a new lead: {lead_name} (Score: {score}). Please follow up within {sla_hours} hours.",
                priority=NotificationPriority.HIGH
            ),
            NotificationTemplate(
                template_id="task_overdue",
                name="Task Overdue",
                notification_type=NotificationType.EMAIL,
                subject_template="Overdue Task: {task_title}",
                body_template="Your task '{task_title}' is now {days_overdue} days overdue. Please complete it as soon as possible.",
                priority=NotificationPriority.URGENT
            ),
            NotificationTemplate(
                template_id="workflow_completed",
                name="Workflow Completed",
                notification_type=NotificationType.SLACK,
                subject_template="Workflow Completed",
                body_template="Workflow '{workflow_name}' has been completed for lead {lead_name}.",
                priority=NotificationPriority.MEDIUM
            ),
            NotificationTemplate(
                template_id="high_score_lead",
                name="High Score Lead Alert",
                notification_type=NotificationType.SMS,
                subject_template="High Score Lead",
                body_template="URGENT: High-score lead {lead_name} (Score: {score}) requires immediate attention!",
                priority=NotificationPriority.URGENT
            )
        ]
        
        for template in templates:
            self.templates[template.template_id] = template
    
    def register_channel(self, notification_type: NotificationType, channel: NotificationChannel):
        """Register a notification channel"""
        self.channels[notification_type] = channel
        logging.info(f"Registered {notification_type.value} channel")
    
    def add_template(self, template: NotificationTemplate):
        """Add a notification template"""
        self.templates[template.template_id] = template
        logging.info(f"Added template: {template.name}")
    
    async def send_notification(self, notification_type: NotificationType,
                               recipient: str, subject: str, body: str,
                               priority: NotificationPriority = NotificationPriority.MEDIUM,
                               metadata: Dict[str, Any] = None) -> str:
        """Send a notification"""
        
        notification_id = f"notif_{uuid.uuid4().hex[:8]}"
        
        notification = Notification(
            notification_id=notification_id,
            notification_type=notification_type,
            recipient=recipient,
            subject=subject,
            body=body,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.notifications[notification_id] = notification
        
        # Send notification asynchronously
        asyncio.create_task(self._process_notification(notification))
        
        return notification_id
    
    async def send_from_template(self, template_id: str, recipient: str,
                                template_data: Dict[str, Any],
                                priority: Optional[NotificationPriority] = None) -> str:
        """Send notification using template"""
        
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        
        # Format subject and body
        subject = template.subject_template.format(**template_data)
        body = template.body_template.format(**template_data)
        
        return await self.send_notification(
            notification_type=template.notification_type,
            recipient=recipient,
            subject=subject,
            body=body,
            priority=priority or template.priority,
            metadata={'template_id': template_id, 'template_data': template_data}
        )
    
    async def _process_notification(self, notification: Notification):
        """Process and send notification"""
        try:
            channel = self.channels.get(notification.notification_type)
            
            if not channel:
                logging.error(f"No channel registered for {notification.notification_type.value}")
                notification.status = NotificationStatus.FAILED
                return
            
            # Validate recipient
            if not channel.validate_recipient(notification.recipient):
                logging.error(f"Invalid recipient format: {notification.recipient}")
                notification.status = NotificationStatus.FAILED
                return
            
            # Attempt to send
            success = await channel.send(notification)
            
            if success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now()
                logging.info(f"Notification {notification.notification_id} sent successfully")
            else:
                await self._handle_send_failure(notification)
                
        except Exception as e:
            logging.error(f"Error processing notification {notification.notification_id}: {str(e)}")
            await self._handle_send_failure(notification)
    
    async def _handle_send_failure(self, notification: Notification):
        """Handle notification send failure with retry logic"""
        notification.retry_count += 1
        
        if notification.retry_count <= notification.max_retries:
            # Exponential backoff
            delay = 2 ** notification.retry_count
            logging.info(f"Retrying notification {notification.notification_id} in {delay} seconds")
            
            await asyncio.sleep(delay)
            await self._process_notification(notification)
        else:
            notification.status = NotificationStatus.FAILED
            logging.error(f"Notification {notification.notification_id} failed after {notification.max_retries} retries")
    
    def subscribe_to_events(self, event_type: str, callback: Callable):
        """Subscribe to notification events"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        logging.info(f"Subscribed to {event_type} events")
    
    async def notify_event(self, event_type: str, event_data: Dict[str, Any]):
        """Notify subscribers of an event"""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(event_data)
                except Exception as e:
                    logging.error(f"Error in event callback: {str(e)}")
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        total_notifications = len(self.notifications)
        
        if total_notifications == 0:
            return {"message": "No notifications sent yet"}
        
        # Status distribution
        status_counts = {}
        type_counts = {}
        priority_counts = {}
        
        for notification in self.notifications.values():
            status = notification.status.value
            ntype = notification.notification_type.value
            priority = notification.priority.name
            
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[ntype] = type_counts.get(ntype, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Success rate
        sent_count = status_counts.get('sent', 0) + status_counts.get('delivered', 0)
        success_rate = sent_count / total_notifications if total_notifications > 0 else 0
        
        return {
            "total_notifications": total_notifications,
            "success_rate": success_rate,
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "priority_distribution": priority_counts,
            "registered_channels": list(self.channels.keys()),
            "available_templates": len(self.templates)
        }
    
    async def send_bulk_notifications(self, notifications_data: List[Dict[str, Any]]) -> List[str]:
        """Send multiple notifications in bulk"""
        notification_ids = []
        
        for notif_data in notifications_data:
            notification_id = await self.send_notification(
                notification_type=NotificationType(notif_data['type']),
                recipient=notif_data['recipient'],
                subject=notif_data['subject'],
                body=notif_data['body'],
                priority=NotificationPriority(notif_data.get('priority', 2)),
                metadata=notif_data.get('metadata', {})
            )
            notification_ids.append(notification_id)
        
        logging.info(f"Sent {len(notification_ids)} bulk notifications")
        return notification_ids

# Alert System for Automation Events
class AutomationAlertSystem:
    """Specialized alert system for automation events"""
    
    def __init__(self, notification_system: NotificationSystem):
        self.notification_system = notification_system
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default automation alerts"""
        self.alert_rules = {
            "high_score_lead": {
                "condition": lambda data: data.get('overall_score', 0) >= 80,
                "template_id": "high_score_lead",
                "recipients": ["sales_manager@company.com"],
                "notification_type": NotificationType.SMS
            },
            "task_overdue": {
                "condition": lambda data: data.get('days_overdue', 0) > 0,
                "template_id": "task_overdue",
                "recipients": ["assignee"],  # Will be replaced with actual assignee
                "notification_type": NotificationType.EMAIL
            },
            "workflow_failure": {
                "condition": lambda data: data.get('status') == 'failed',
                "template_id": "workflow_completed",
                "recipients": ["admin@company.com"],
                "notification_type": NotificationType.SLACK
            }
        }
    
    async def check_and_send_alerts(self, event_type: str, event_data: Dict[str, Any]):
        """Check conditions and send alerts if needed"""
        
        if event_type in self.alert_rules:
            rule = self.alert_rules[event_type]
            
            # Check condition
            if rule["condition"](event_data):
                # Send alerts to all recipients
                for recipient in rule["recipients"]:
                    # Replace placeholder recipients
                    if recipient == "assignee":
                        recipient = event_data.get('assignee_email', 'unknown@company.com')
                    
                    await self.notification_system.send_from_template(
                        template_id=rule["template_id"],
                        recipient=recipient,
                        template_data=event_data
                    )
                
                logging.info(f"Sent {event_type} alerts to {len(rule['recipients'])} recipients")

# Global notification system instance
notification_system = NotificationSystem()
alert_system = AutomationAlertSystem(notification_system)

# Initialize channels (would be configured with actual credentials)
notification_system.register_channel(
    NotificationType.EMAIL, 
    EmailChannel({"smtp_server": "localhost", "port": 587})
)
notification_system.register_channel(
    NotificationType.SMS, 
    SMSChannel({"api_key": "your_sms_api_key"})
)
notification_system.register_channel(
    NotificationType.SLACK, 
    SlackChannel({"webhook_url": "your_slack_webhook"})
)
notification_system.register_channel(
    NotificationType.WEBHOOK, 
    WebhookChannel({"timeout": 30})
)