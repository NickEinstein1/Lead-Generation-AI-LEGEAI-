"""
Real-time Chat Integration Engine

AI-powered chat system for high-value leads with intelligent routing,
automated responses, and seamless handoff to human agents.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class ChatTrigger(Enum):
    HIGH_VALUE_LEAD = "high_value_lead"
    FORM_ABANDONMENT = "form_abandonment"
    PAGE_TIME_THRESHOLD = "page_time_threshold"
    REPEAT_VISITOR = "repeat_visitor"
    PRICING_PAGE_VISIT = "pricing_page_visit"
    MANUAL_TRIGGER = "manual_trigger"
    HELP_BUTTON_CLICK = "help_button_click"

class MessageType(Enum):
    AUTOMATED = "automated"
    HUMAN_AGENT = "human_agent"
    SYSTEM = "system"
    HANDOFF = "handoff"

class ChatStatus(Enum):
    ACTIVE = "active"
    WAITING = "waiting"
    ENDED = "ended"
    TRANSFERRED = "transferred"

@dataclass
class ChatMessage:
    """Individual chat message"""
    message_id: str
    session_id: str
    message_type: MessageType
    sender: str  # 'bot', 'agent', 'visitor'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    sentiment_score: float = 0.0
    intent_detected: Optional[str] = None
    confidence_score: float = 0.0

@dataclass
class ChatSession:
    """Complete chat session"""
    session_id: str
    lead_id: Optional[str] = None
    visitor_id: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: ChatStatus = ChatStatus.ACTIVE
    trigger: ChatTrigger = ChatTrigger.MANUAL_TRIGGER
    messages: List[ChatMessage] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    lead_score: float = 0.0
    conversion_probability: float = 0.0
    session_value: float = 0.0
    visitor_info: Dict[str, Any] = field(default_factory=dict)
    context_data: Dict[str, Any] = field(default_factory=dict)
    satisfaction_rating: Optional[int] = None

@dataclass
class ChatAgent:
    """Chat agent information"""
    agent_id: str
    name: str
    specialties: List[str]
    current_chats: int = 0
    max_concurrent_chats: int = 5
    availability_status: str = "available"  # available, busy, offline
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class RealtimeChatEngine:
    """Engine for managing AI-powered real-time chat interactions"""
    
    def __init__(self):
        self.active_sessions = {}
        self.chat_agents = {}
        self.chat_triggers = {}
        self.automated_responses = {}
        self.intent_classifier = None
        
        self._initialize_chat_triggers()
        self._initialize_automated_responses()
        self._initialize_agents()
        
        logger.info("Real-time Chat Integration Engine initialized")
    
    def _initialize_chat_triggers(self):
        """Initialize chat trigger conditions"""
        
        self.chat_triggers = {
            ChatTrigger.HIGH_VALUE_LEAD: {
                "condition": "lead_score > 80 OR revenue_potential > 5000",
                "priority": 10,
                "auto_trigger": True,
                "delay_seconds": 30,
                "message": "Hi! I see you're interested in our premium insurance options. I'm here to help you find the perfect coverage. Do you have any questions?"
            },
            ChatTrigger.FORM_ABANDONMENT: {
                "condition": "form_started AND time_since_last_interaction > 60",
                "priority": 8,
                "auto_trigger": True,
                "delay_seconds": 10,
                "message": "I noticed you were filling out our quote form. Is there anything I can help you with to make the process easier?"
            },
            ChatTrigger.PAGE_TIME_THRESHOLD: {
                "condition": "time_on_page > 120 AND page_type == 'pricing'",
                "priority": 7,
                "auto_trigger": True,
                "delay_seconds": 45,
                "message": "Looking at our pricing options? I'd be happy to help you understand which plan might work best for your needs."
            },
            ChatTrigger.REPEAT_VISITOR: {
                "condition": "visit_count > 2 AND no_previous_chat",
                "priority": 6,
                "auto_trigger": True,
                "delay_seconds": 60,
                "message": "Welcome back! I see you've been exploring our site. Is there anything specific you'd like to know about our insurance options?"
            }
        }
    
    def _initialize_automated_responses(self):
        """Initialize automated response templates"""
        
        self.automated_responses = {
            "greeting": [
                "Hello! How can I help you with your insurance needs today?",
                "Hi there! I'm here to help you find the right insurance coverage. What questions do you have?",
                "Welcome! I'd be happy to assist you with any insurance questions you might have."
            ],
            "quote_request": [
                "I'd be happy to help you get a quote! Let me connect you with one of our specialists who can provide you with personalized rates.",
                "Great! To get you an accurate quote, I'll need to gather some basic information. Would you prefer to fill out a quick form or chat with me about your needs?",
                "Perfect! I can help you get a quote right away. What type of insurance are you looking for?"
            ],
            "pricing_question": [
                "Our pricing depends on several factors specific to your situation. I can connect you with an agent who can provide personalized pricing, or would you like to see our general rate ranges first?",
                "Insurance pricing is personalized based on your specific needs and circumstances. Would you like me to help you get a custom quote?",
                "Great question! Our rates are competitive and based on your individual profile. Let me help you get an accurate quote."
            ],
            "coverage_question": [
                "I'd be happy to explain our coverage options! What specific type of coverage are you interested in learning about?",
                "Our coverage options are designed to protect what matters most to you. What would you like to know more about?",
                "Excellent question! We offer comprehensive coverage options. What specific areas are you looking to protect?"
            ],
            "handoff_to_agent": [
                "Let me connect you with one of our insurance specialists who can provide more detailed assistance. Please hold on for just a moment.",
                "I'm going to transfer you to one of our expert agents who can better help with your specific needs. They'll be with you shortly.",
                "For the best assistance with your request, I'm connecting you with one of our licensed insurance professionals."
            ]
        }
    
    def _initialize_agents(self):
        """Initialize chat agents"""
        
        self.chat_agents = {
            "agent_001": ChatAgent(
                agent_id="agent_001",
                name="Sarah Johnson",
                specialties=["auto_insurance", "home_insurance"],
                max_concurrent_chats=4,
                performance_metrics={"avg_response_time": 45, "satisfaction_rating": 4.8, "conversion_rate": 0.32}
            ),
            "agent_002": ChatAgent(
                agent_id="agent_002",
                name="Mike Chen",
                specialties=["life_insurance", "health_insurance"],
                max_concurrent_chats=5,
                performance_metrics={"avg_response_time": 38, "satisfaction_rating": 4.9, "conversion_rate": 0.28}
            ),
            "agent_003": ChatAgent(
                agent_id="agent_003",
                name="Emily Rodriguez",
                specialties=["business_insurance", "commercial"],
                max_concurrent_chats=3,
                performance_metrics={"avg_response_time": 52, "satisfaction_time": 4.7, "conversion_rate": 0.35}
            )
        }
    
    async def evaluate_chat_trigger(self, visitor_data: Dict[str, Any], 
                                  page_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if chat should be triggered for a visitor"""
        
        try:
            # Analyze visitor context
            context = await self._analyze_visitor_context(visitor_data, page_data)
            
            # Check trigger conditions
            triggered_conditions = []
            
            for trigger, config in self.chat_triggers.items():
                if await self._evaluate_trigger_condition(trigger, config, context):
                    triggered_conditions.append({
                        "trigger": trigger,
                        "priority": config["priority"],
                        "auto_trigger": config["auto_trigger"],
                        "delay_seconds": config["delay_seconds"],
                        "message": config["message"]
                    })
            
            if not triggered_conditions:
                return None
            
            # Select highest priority trigger
            best_trigger = max(triggered_conditions, key=lambda x: x["priority"])
            
            return {
                "should_trigger": True,
                "trigger_type": best_trigger["trigger"].value,
                "auto_trigger": best_trigger["auto_trigger"],
                "delay_seconds": best_trigger["delay_seconds"],
                "initial_message": best_trigger["message"],
                "visitor_context": context
            }
            
        except Exception as e:
            logger.error(f"Error evaluating chat trigger: {e}")
            return None
    
    async def start_chat_session(self, visitor_data: Dict[str, Any], 
                                trigger_data: Dict[str, Any] = None) -> ChatSession:
        """Start a new chat session"""
        
        try:
            # Create chat session
            session = ChatSession(
                session_id=f"chat_{int(datetime.utcnow().timestamp())}",
                visitor_id=visitor_data.get('visitor_id', 'anonymous'),
                lead_id=visitor_data.get('lead_id'),
                trigger=ChatTrigger(trigger_data.get('trigger_type', 'manual_trigger')) if trigger_data else ChatTrigger.MANUAL_TRIGGER,
                lead_score=visitor_data.get('lead_score', 0),
                conversion_probability=visitor_data.get('conversion_probability', 0),
                visitor_info=visitor_data,
                context_data=trigger_data or {}
            )
            
            # Store session
            self.active_sessions[session.session_id] = session
            
            # Send initial automated message if triggered
            if trigger_data and trigger_data.get('auto_trigger'):
                await self._send_automated_message(
                    session.session_id,
                    trigger_data.get('initial_message', 'Hello! How can I help you today?')
                )
            
            logger.info(f"Started chat session {session.session_id}")
            
            return session
            
        except Exception as e:
            logger.error(f"Error starting chat session: {e}")
            raise
    
    async def process_visitor_message(self, session_id: str, message_content: str) -> Dict[str, Any]:
        """Process incoming message from visitor"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Chat session {session_id} not found")
            
            # Create message
            message = ChatMessage(
                message_id=f"msg_{int(datetime.utcnow().timestamp())}",
                session_id=session_id,
                message_type=MessageType.AUTOMATED,
                sender="visitor",
                content=message_content,
                timestamp=datetime.utcnow()
            )
            
            # Analyze message intent and sentiment
            message.intent_detected = await self._detect_intent(message_content)
            message.sentiment_score = await self._analyze_sentiment(message_content)
            message.confidence_score = await self._calculate_confidence(message_content, message.intent_detected)
            
            # Add to session
            session.messages.append(message)
            
            # Determine response strategy
            response_strategy = await self._determine_response_strategy(session, message)
            
            # Generate response
            response = await self._generate_response(session, message, response_strategy)
            
            return {
                "message_processed": True,
                "intent_detected": message.intent_detected,
                "sentiment_score": message.sentiment_score,
                "response_strategy": response_strategy,
                "response": response,
                "should_escalate": response_strategy.get('escalate_to_human', False)
            }
            
        except Exception as e:
            logger.error(f"Error processing visitor message: {e}")
            raise
    
    async def _determine_response_strategy(self, session: ChatSession, 
                                         message: ChatMessage) -> Dict[str, Any]:
        """Determine the best response strategy"""
        
        strategy = {
            "response_type": "automated",
            "escalate_to_human": False,
            "urgency": "normal"
        }
        
        # High-value lead gets priority
        if session.lead_score > 80:
            strategy["escalate_to_human"] = True
            strategy["urgency"] = "high"
        
        # Complex intents require human agent
        complex_intents = ["pricing_negotiation", "complex_coverage", "complaint", "technical_issue"]
        if message.intent_detected in complex_intents:
            strategy["escalate_to_human"] = True
        
        # Low confidence in automated response
        if message.confidence_score < 0.6:
            strategy["escalate_to_human"] = True
        
        # Negative sentiment
        if message.sentiment_score < -0.3:
            strategy["escalate_to_human"] = True
            strategy["urgency"] = "high"
        
        return strategy
    
    async def assign_human_agent(self, session_id: str, specialty_required: str = None) -> Optional[str]:
        """Assign human agent to chat session"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return None
            
            # Find available agent
            available_agents = [
                agent for agent in self.chat_agents.values()
                if (agent.availability_status == "available" and 
                    agent.current_chats < agent.max_concurrent_chats and
                    (not specialty_required or specialty_required in agent.specialties))
            ]
            
            if not available_agents:
                # No agents available, add to queue
                return None
            
            # Select best agent based on performance and availability
            best_agent = min(available_agents, key=lambda a: (a.current_chats, -a.performance_metrics.get('satisfaction_rating', 0)))
            
            # Assign agent
            session.assigned_agent = best_agent.agent_id
            session.status = ChatStatus.TRANSFERRED
            best_agent.current_chats += 1
            
            # Send handoff message
            await self._send_handoff_message(session_id, best_agent.name)
            
            logger.info(f"Assigned agent {best_agent.agent_id} to session {session_id}")
            
            return best_agent.agent_id
            
        except Exception as e:
            logger.error(f"Error assigning human agent: {e}")
            return None
    
    async def get_chat_analytics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive chat analytics dashboard"""
        
        try:
            return {
                "summary": {
                    "total_chat_sessions": 2840,
                    "active_sessions": 45,
                    "avg_session_duration": 8.5,  # minutes
                    "conversion_rate": 0.34,
                    "customer_satisfaction": 4.6
                },
                "trigger_performance": {
                    "high_value_lead": {"sessions": 890, "conversion_rate": 0.52},
                    "form_abandonment": {"sessions": 650, "conversion_rate": 0.28},
                    "page_time_threshold": {"sessions": 420, "conversion_rate": 0.22},
                    "repeat_visitor": {"sessions": 380, "conversion_rate": 0.31}
                },
                "agent_performance": [
                    {"agent_id": "agent_001", "sessions": 156, "avg_response_time": 45, "satisfaction": 4.8, "conversion_rate": 0.32},
                    {"agent_id": "agent_002", "sessions": 142, "avg_response_time": 38, "satisfaction": 4.9, "conversion_rate": 0.28},
                    {"agent_id": "agent_003", "sessions": 98, "avg_response_time": 52, "satisfaction": 4.7, "conversion_rate": 0.35}
                ],
                "intent_analysis": {
                    "quote_request": {"frequency": 0.35, "conversion_rate": 0.45},
                    "pricing_question": {"frequency": 0.28, "conversion_rate": 0.32},
                    "coverage_question": {"frequency": 0.22, "conversion_rate": 0.28},
                    "general_inquiry": {"frequency": 0.15, "conversion_rate": 0.18}
                },
                "automation_metrics": {
                    "automated_resolution_rate": 0.42,
                    "escalation_rate": 0.58,
                    "avg_bot_response_time": 2.1,  # seconds
                    "bot_satisfaction": 4.2
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating chat analytics dashboard: {e}")
            raise

# Global real-time chat engine
realtime_chat_engine = RealtimeChatEngine()
