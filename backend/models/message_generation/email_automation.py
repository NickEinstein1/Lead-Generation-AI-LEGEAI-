import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import asyncio
import aiosmtplib

class EmailAutomationSystem:
    """
    Automated email communication system for leads with verified emails
    """
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
        self.sent_emails = []
        self.email_templates = self._load_email_templates()
        
    async def send_personalized_email(self, lead_data: Dict[str, Any],
                                    scoring_result: Dict[str, Any],
                                    message_content: Dict[str, Any],
                                    email_type: str = "initial_contact") -> Dict[str, Any]:
        """
        Send personalized email to lead with verified email
        """
        
        # Verify email exists and is valid
        email = lead_data.get('email')
        if not email or not self._is_valid_email(email):
            return {
                "status": "failed",
                "reason": "Invalid or missing email address",
                "email": email
            }
        
        # Check if email is verified/accessible
        email_verification = lead_data.get('email_verification', {})
        if not email_verification.get('verified', False):
            return {
                "status": "failed",
                "reason": "Email not verified or accessible",
                "email": email
            }
        
        try:
            # Create email message
            email_message = self._create_email_message(
                lead_data, scoring_result, message_content, email_type
            )
            
            # Send email
            result = await self._send_email_async(email, email_message)
            
            # Log email sent
            self._log_email_sent(lead_data, email_type, result)
            
            # Schedule follow-up if needed
            await self._schedule_follow_up(lead_data, scoring_result, email_type)
            
            return {
                "status": "success",
                "email": email,
                "message_id": result.get('message_id'),
                "sent_at": datetime.now().isoformat(),
                "email_type": email_type,
                "follow_up_scheduled": result.get('follow_up_scheduled', False)
            }
            
        except Exception as e:
            logging.error(f"Error sending email to {email}: {str(e)}")
            return {
                "status": "failed",
                "reason": str(e),
                "email": email
            }
    
    async def send_bulk_emails(self, leads_with_emails: List[Dict[str, Any]],
                             email_type: str = "bulk_campaign") -> Dict[str, Any]:
        """
        Send bulk emails to multiple leads with rate limiting
        """
        
        results = []
        successful_sends = 0
        failed_sends = 0
        
        # Rate limiting: 50 emails per minute
        batch_size = 50
        delay_between_batches = 60  # seconds
        
        for i in range(0, len(leads_with_emails), batch_size):
            batch = leads_with_emails[i:i + batch_size]
            
            # Process batch
            batch_tasks = []
            for lead_data in batch:
                # Generate message for this lead
                message_content = self._generate_bulk_message_content(lead_data, email_type)
                
                # Create send task
                task = self.send_personalized_email(
                    lead_data, 
                    lead_data.get('scoring_result', {}), 
                    message_content, 
                    email_type
                )
                batch_tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    failed_sends += 1
                    results.append({"status": "failed", "reason": str(result)})
                else:
                    if result.get('status') == 'success':
                        successful_sends += 1
                    else:
                        failed_sends += 1
                    results.append(result)
            
            # Wait before next batch (except for last batch)
            if i + batch_size < len(leads_with_emails):
                await asyncio.sleep(delay_between_batches)
        
        return {
            "total_leads": len(leads_with_emails),
            "successful_sends": successful_sends,
            "failed_sends": failed_sends,
            "success_rate": successful_sends / len(leads_with_emails) if leads_with_emails else 0,
            "results": results,
            "campaign_type": email_type,
            "completed_at": datetime.now().isoformat()
        }
    
    async def send_sequence_emails(self, lead_data: Dict[str, Any],
                                 sequence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send automated email sequence based on lead scoring
        """
        
        sequence_results = []
        lead_id = lead_data.get('lead_id')
        
        for step in sequence.get('sequence', []):
            if step.get('message', {}).get('type') == 'email':
                
                # Calculate send time
                timing = step.get('timing', 'immediate')
                send_time = self._calculate_send_time(timing)
                
                # Schedule email
                if timing == 'immediate':
                    # Send immediately
                    result = await self.send_personalized_email(
                        lead_data,
                        lead_data.get('scoring_result', {}),
                        step.get('message', {}),
                        f"sequence_step_{step.get('step')}"
                    )
                else:
                    # Schedule for later
                    result = await self._schedule_email(
                        lead_data, step.get('message', {}), send_time
                    )
                
                sequence_results.append({
                    "step": step.get('step'),
                    "timing": timing,
                    "send_time": send_time.isoformat() if hasattr(send_time, 'isoformat') else send_time,
                    "result": result
                })
        
        return {
            "lead_id": lead_id,
            "sequence_type": sequence.get('sequence_type'),
            "total_steps": len(sequence_results),
            "steps_completed": len([r for r in sequence_results if r['result']['status'] == 'success']),
            "sequence_results": sequence_results
        }
    
    def _create_email_message(self, lead_data: Dict[str, Any],
                            scoring_result: Dict[str, Any],
                            message_content: Dict[str, Any],
                            email_type: str) -> Dict[str, Any]:
        """
        Create formatted email message
        """
        
        name = lead_data.get('name', 'there')
        email = lead_data.get('email')
        
        # Get message content
        subject = message_content.get('subject_line', f"Important information for {name}")
        content = message_content.get('content', '')
        cta = message_content.get('call_to_action', '')
        
        # Create HTML email
        html_content = self._create_html_email(name, content, cta, email_type, scoring_result)
        
        # Create plain text version
        text_content = self._create_text_email(name, content, cta)
        
        return {
            "to_email": email,
            "to_name": name,
            "subject": subject,
            "html_content": html_content,
            "text_content": text_content,
            "email_type": email_type,
            "personalization_data": {
                "name": name,
                "priority": scoring_result.get('priority_level'),
                "products": scoring_result.get('recommended_products', [])
            }
        }
    
    def _create_html_email(self, name: str, content: str, cta: str, 
                         email_type: str, scoring_result: Dict[str, Any]) -> str:
        """
        Create HTML email template
        """
        
        priority = scoring_result.get('priority_level', 'MEDIUM')
        primary_product = scoring_result.get('recommended_products', ['insurance'])[0]
        
        # Color scheme based on priority
        color_scheme = {
            'CRITICAL': {'primary': '#FF4444', 'secondary': '#FFE6E6'},
            'HIGH': {'primary': '#FF8800', 'secondary': '#FFF4E6'},
            'MEDIUM': {'primary': '#0066CC', 'secondary': '#E6F3FF'},
            'LOW': {'primary': '#00AA44', 'secondary': '#E6FFE6'}
        }
        
        colors = color_scheme.get(priority, color_scheme['MEDIUM'])
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Insurance Information for {name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {colors['primary']}; color: white; padding: 20px; text-align: center; }}
                .content {{ background-color: {colors['secondary']}; padding: 30px; }}
                .cta-button {{ 
                    display: inline-block; 
                    background-color: {colors['primary']}; 
                    color: white; 
                    padding: 12px 24px; 
                    text-decoration: none; 
                    border-radius: 5px; 
                    margin: 20px 0;
                }}
                .footer {{ background-color: #f4f4f4; padding: 20px; text-align: center; font-size: 12px; }}
                .priority-badge {{ 
                    background-color: {colors['primary']}; 
                    color: white; 
                    padding: 5px 10px; 
                    border-radius: 15px; 
                    font-size: 12px; 
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🛡️ Your {primary_product.title()} Insurance Information</h1>
                    <span class="priority-badge">{priority} Priority</span>
                </div>
                
                <div class="content">
                    {content.replace(chr(10), '<br>')}
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="#" class="cta-button">{cta}</a>
                    </div>
                    
                    <div style="background-color: white; padding: 15px; border-left: 4px solid {colors['primary']}; margin: 20px 0;">
                        <h3>Why Choose Us?</h3>
                        <ul>
                            <li>✅ A+ Rated Insurance Company</li>
                            <li>✅ 24/7 Customer Support</li>
                            <li>✅ Claims Processed in 24 Hours</li>
                            <li>✅ Over 50,000 Satisfied Customers</li>
                        </ul>
                    </div>
                </div>
                
                <div class="footer">
                    <p>This email was sent because you expressed interest in {primary_product} insurance.</p>
                    <p>📞 Call us: 1-800-INSURANCE | 📧 Email: info@company.com</p>
                    <p><a href="#">Unsubscribe</a> | <a href="#">Privacy Policy</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _create_text_email(self, name: str, content: str, cta: str) -> str:
        """
        Create plain text email version
        """
        
        text_email = f"""
        Hi {name},

        {content}

        {cta}

        Why Choose Us?
        - A+ Rated Insurance Company
        - 24/7 Customer Support  
        - Claims Processed in 24 Hours
        - Over 50,000 Satisfied Customers

        Best regards,
        [Agent Name]
        [Company Name]

        📞 1-800-INSURANCE
        📧 info@company.com

        To unsubscribe, reply with "UNSUBSCRIBE"
        """
        
        return text_email
    
    async def _send_email_async(self, to_email: str, email_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send email asynchronously using aiosmtplib
        """
        
        # Create MIME message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = email_message['subject']
        msg['From'] = self.smtp_config['from_email']
        msg['To'] = to_email
        
        # Add text and HTML parts
        text_part = MIMEText(email_message['text_content'], 'plain')
        html_part = MIMEText(email_message['html_content'], 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        try:
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_config['smtp_server'],
                port=self.smtp_config['smtp_port'],
                username=self.smtp_config['username'],
                password=self.smtp_config['password'],
                use_tls=True
            )
            
            return {
                "status": "success",
                "message_id": f"msg_{datetime.now().timestamp()}",
                "sent_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"SMTP Error: {str(e)}")
    
    def _is_valid_email(self, email: str) -> bool:
        """
        Validate email format
        """
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _calculate_send_time(self, timing: str) -> datetime:
        """
        Calculate when to send email based on timing
        """
        now = datetime.now()
        
        timing_map = {
            'immediate': now,
            '2_hours': now + timedelta(hours=2),
            '4_hours': now + timedelta(hours=4),
            '1_day': now + timedelta(days=1),
            '2_days': now + timedelta(days=2),
            '1_week': now + timedelta(weeks=1),
            '2_weeks': now + timedelta(weeks=2),
            '3_weeks': now + timedelta(weeks=3),
            '1_month': now + timedelta(days=30)
        }
        
        return timing_map.get(timing, now)
    
    async def _schedule_email(self, lead_data: Dict[str, Any], 
                            message_content: Dict[str, Any], 
                            send_time: datetime) -> Dict[str, Any]:
        """
        Schedule email for future sending
        """
        
        # In a real implementation, this would use a task queue like Celery
        # For now, we'll just return a scheduled status
        
        return {
            "status": "scheduled",
            "scheduled_time": send_time.isoformat(),
            "lead_id": lead_data.get('lead_id'),
            "message_type": "scheduled_email"
        }
    
    async def _schedule_follow_up(self, lead_data: Dict[str, Any],
                                scoring_result: Dict[str, Any],
                                email_type: str) -> bool:
        """
        Schedule follow-up email based on lead priority
        """
        
        priority = scoring_result.get('priority_level', 'MEDIUM')
        
        # Follow-up timing based on priority
        follow_up_timing = {
            'CRITICAL': timedelta(hours=4),
            'HIGH': timedelta(hours=24),
            'MEDIUM': timedelta(days=3),
            'LOW': timedelta(weeks=1)
        }
        
        follow_up_time = datetime.now() + follow_up_timing.get(priority, timedelta(days=3))
        
        # Schedule follow-up (would integrate with task queue)
        logging.info(f"Follow-up scheduled for {lead_data.get('email')} at {follow_up_time}")
        
        return True
    
    def _generate_bulk_message_content(self, lead_data: Dict[str, Any], email_type: str) -> Dict[str, Any]:
        """
        Generate message content for bulk campaigns
        """
        
        name = lead_data.get('name', 'there')
        
        if email_type == "bulk_campaign":
            return {
                "subject_line": f"{name}, exclusive insurance savings inside!",
                "content": f"Hi {name},\n\nWe have exclusive insurance offers available in your area. Save up to 40% on your premiums while getting better coverage.\n\nDon't miss out on these limited-time savings!",
                "call_to_action": "Get Your Free Quote Now"
            }
        
        return {
            "subject_line": f"Important information for {name}",
            "content": f"Hi {name},\n\nWe have important insurance information for you.",
            "call_to_action": "Learn More"
        }
    
    def _log_email_sent(self, lead_data: Dict[str, Any], email_type: str, result: Dict[str, Any]):
        """
        Log email sending for analytics
        """
        
        log_entry = {
            "lead_id": lead_data.get('lead_id'),
            "email": lead_data.get('email'),
            "email_type": email_type,
            "sent_at": datetime.now().isoformat(),
            "status": result.get('status'),
            "message_id": result.get('message_id')
        }
        
        self.sent_emails.append(log_entry)
        logging.info(f"Email logged: {log_entry}")
    
    def _load_email_templates(self) -> Dict[str, Any]:
        """
        Load email templates from database/files
        """
        
        return {
            "initial_contact": {},
            "follow_up": {},
            "urgent": {},
            "educational": {},
            "promotional": {}
        }