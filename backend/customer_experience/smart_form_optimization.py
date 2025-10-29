"""
Smart Form Optimization Engine

Adaptive forms that dynamically adjust based on user behavior,
device type, and conversion optimization to maximize completion rates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class FieldType(Enum):
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DATE = "date"
    NUMBER = "number"
    TEXTAREA = "textarea"
    FILE_UPLOAD = "file_upload"

class ValidationRule(Enum):
    REQUIRED = "required"
    EMAIL_FORMAT = "email_format"
    PHONE_FORMAT = "phone_format"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    NUMERIC_ONLY = "numeric_only"
    CUSTOM_REGEX = "custom_regex"

@dataclass
class FormField:
    """Individual form field configuration"""
    field_id: str
    field_type: FieldType
    label: str
    placeholder: str
    required: bool = False
    validation_rules: List[ValidationRule] = field(default_factory=list)
    options: List[str] = field(default_factory=list)  # For select/radio fields
    conditional_logic: Dict[str, Any] = field(default_factory=dict)
    completion_rate: float = 0.0
    abandonment_rate: float = 0.0
    error_rate: float = 0.0
    avg_completion_time: float = 0.0
    priority_score: float = 0.0

@dataclass
class FormConfiguration:
    """Complete form configuration"""
    form_id: str
    form_name: str
    form_type: str
    fields: List[FormField]
    multi_step: bool = False
    step_configuration: List[List[str]] = field(default_factory=list)  # Field IDs per step
    completion_rate: float = 0.0
    abandonment_points: Dict[str, float] = field(default_factory=dict)
    conversion_rate: float = 0.0
    avg_completion_time: float = 0.0
    mobile_optimized: bool = True
    progressive_profiling: bool = False

@dataclass
class FormSession:
    """Individual form session tracking"""
    session_id: str
    form_id: str
    lead_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    completed: bool = False
    current_step: int = 1
    field_interactions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    abandonment_point: Optional[str] = None
    device_type: str = "desktop"
    user_agent: str = ""
    errors_encountered: List[Dict[str, Any]] = field(default_factory=list)

class SmartFormOptimizationEngine:
    """Engine for optimizing form conversion rates through adaptive design"""
    
    def __init__(self):
        self.form_configurations = {}
        self.active_sessions = {}
        self.optimization_rules = {}
        self.ab_test_variants = {}
        
        self._initialize_form_templates()
        self._initialize_optimization_rules()
        
        logger.info("Smart Form Optimization Engine initialized")
    
    def _initialize_form_templates(self):
        """Initialize form templates for different use cases"""
        
        # Insurance quote form
        quote_form_fields = [
            FormField(
                field_id="first_name",
                field_type=FieldType.TEXT,
                label="First Name",
                placeholder="Enter your first name",
                required=True,
                priority_score=10.0
            ),
            FormField(
                field_id="last_name",
                field_type=FieldType.TEXT,
                label="Last Name",
                placeholder="Enter your last name",
                required=True,
                priority_score=9.0
            ),
            FormField(
                field_id="email",
                field_type=FieldType.EMAIL,
                label="Email Address",
                placeholder="your.email@example.com",
                required=True,
                validation_rules=[ValidationRule.EMAIL_FORMAT],
                priority_score=10.0
            ),
            FormField(
                field_id="phone",
                field_type=FieldType.PHONE,
                label="Phone Number",
                placeholder="(555) 123-4567",
                required=False,
                validation_rules=[ValidationRule.PHONE_FORMAT],
                priority_score=7.0
            ),
            FormField(
                field_id="zip_code",
                field_type=FieldType.TEXT,
                label="ZIP Code",
                placeholder="12345",
                required=True,
                validation_rules=[ValidationRule.NUMERIC_ONLY, ValidationRule.MIN_LENGTH],
                priority_score=8.0
            ),
            FormField(
                field_id="insurance_type",
                field_type=FieldType.SELECT,
                label="Insurance Type",
                placeholder="Select insurance type",
                required=True,
                options=["Auto Insurance", "Home Insurance", "Life Insurance", "Health Insurance"],
                priority_score=9.0
            ),
            FormField(
                field_id="current_provider",
                field_type=FieldType.SELECT,
                label="Current Insurance Provider",
                placeholder="Select your current provider",
                required=False,
                options=["State Farm", "Geico", "Progressive", "Allstate", "Other", "No Current Coverage"],
                priority_score=5.0
            ),
            FormField(
                field_id="coverage_amount",
                field_type=FieldType.SELECT,
                label="Desired Coverage Amount",
                placeholder="Select coverage amount",
                required=False,
                options=["$100,000", "$250,000", "$500,000", "$1,000,000", "Not Sure"],
                priority_score=6.0
            )
        ]
        
        self.form_configurations["insurance_quote"] = FormConfiguration(
            form_id="insurance_quote",
            form_name="Insurance Quote Request",
            form_type="lead_generation",
            fields=quote_form_fields,
            multi_step=True,
            step_configuration=[
                ["first_name", "last_name", "email"],  # Step 1: Basic info
                ["phone", "zip_code"],  # Step 2: Contact details
                ["insurance_type", "current_provider", "coverage_amount"]  # Step 3: Insurance details
            ],
            progressive_profiling=True
        )
    
    def _initialize_optimization_rules(self):
        """Initialize form optimization rules"""
        
        self.optimization_rules = {
            "reduce_fields_mobile": {
                "condition": "device_type == 'mobile' AND completion_rate < 0.3",
                "action": "reduce_required_fields",
                "priority": 9
            },
            "progressive_profiling": {
                "condition": "returning_visitor AND has_partial_data",
                "action": "pre_populate_known_fields",
                "priority": 8
            },
            "high_abandonment_field": {
                "condition": "field_abandonment_rate > 0.4",
                "action": "make_field_optional_or_remove",
                "priority": 7
            },
            "slow_completion": {
                "condition": "avg_field_completion_time > 30_seconds",
                "action": "add_field_help_text",
                "priority": 6
            },
            "high_error_rate": {
                "condition": "field_error_rate > 0.2",
                "action": "improve_validation_messaging",
                "priority": 8
            }
        }
    
    async def generate_optimized_form(self, form_type: str, lead_data: Dict[str, Any] = None,
                                    device_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate an optimized form configuration"""
        
        try:
            # Get base form configuration
            base_config = self.form_configurations.get(form_type)
            if not base_config:
                raise ValueError(f"Form type {form_type} not found")
            
            # Analyze context
            context = await self._analyze_form_context(lead_data, device_info)
            
            # Apply optimizations
            optimized_config = await self._apply_optimizations(base_config, context)
            
            # Generate form session
            session = FormSession(
                session_id=f"session_{int(datetime.utcnow().timestamp())}",
                form_id=optimized_config.form_id,
                lead_id=lead_data.get('lead_id') if lead_data else None,
                device_type=device_info.get('device_type', 'desktop') if device_info else 'desktop',
                user_agent=device_info.get('user_agent', '') if device_info else ''
            )
            
            # Store session
            self.active_sessions[session.session_id] = session
            
            # Return optimized form
            return {
                "session_id": session.session_id,
                "form_config": {
                    "form_id": optimized_config.form_id,
                    "form_name": optimized_config.form_name,
                    "multi_step": optimized_config.multi_step,
                    "steps": self._generate_form_steps(optimized_config, context),
                    "progressive_profiling": optimized_config.progressive_profiling
                },
                "optimization_applied": context.get('optimizations_applied', []),
                "predicted_completion_rate": await self._predict_completion_rate(optimized_config, context)
            }
            
        except Exception as e:
            logger.error(f"Error generating optimized form: {e}")
            raise
    
    async def _apply_optimizations(self, base_config: FormConfiguration, 
                                 context: Dict[str, Any]) -> FormConfiguration:
        """Apply optimization rules to form configuration"""
        
        optimized_config = FormConfiguration(
            form_id=base_config.form_id,
            form_name=base_config.form_name,
            form_type=base_config.form_type,
            fields=base_config.fields.copy(),
            multi_step=base_config.multi_step,
            step_configuration=base_config.step_configuration.copy(),
            mobile_optimized=base_config.mobile_optimized,
            progressive_profiling=base_config.progressive_profiling
        )
        
        # Mobile optimization
        if context.get('device_type') == 'mobile':
            optimized_config = await self._optimize_for_mobile(optimized_config)
        
        # Progressive profiling
        if context.get('returning_visitor') and optimized_config.progressive_profiling:
            optimized_config = await self._apply_progressive_profiling(optimized_config, context)
        
        # Field prioritization
        optimized_config = await self._prioritize_fields(optimized_config, context)
        
        # A/B test variants
        if context.get('ab_test_enabled'):
            optimized_config = await self._apply_ab_test_variant(optimized_config, context)
        
        return optimized_config
    
    async def track_field_interaction(self, session_id: str, field_id: str, 
                                    interaction_type: str, interaction_data: Dict[str, Any] = None):
        """Track field-level interactions for optimization"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            if field_id not in session.field_interactions:
                session.field_interactions[field_id] = {
                    'focus_time': None,
                    'blur_time': None,
                    'value_changes': 0,
                    'errors': [],
                    'completed': False
                }
            
            field_data = session.field_interactions[field_id]
            
            if interaction_type == 'focus':
                field_data['focus_time'] = datetime.utcnow().isoformat()
            elif interaction_type == 'blur':
                field_data['blur_time'] = datetime.utcnow().isoformat()
            elif interaction_type == 'change':
                field_data['value_changes'] += 1
            elif interaction_type == 'error':
                field_data['errors'].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'error_type': interaction_data.get('error_type'),
                    'error_message': interaction_data.get('error_message')
                })
            elif interaction_type == 'complete':
                field_data['completed'] = True
            
            logger.debug(f"Tracked {interaction_type} interaction for field {field_id}")
            
        except Exception as e:
            logger.error(f"Error tracking field interaction: {e}")
    
    async def complete_form_session(self, session_id: str, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete form session and analyze performance"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Mark session as completed
            session.completed = True
            session.end_time = datetime.utcnow()
            
            # Calculate session metrics
            session_duration = (session.end_time - session.start_time).total_seconds()
            
            # Analyze form performance
            performance_analysis = await self._analyze_session_performance(session)
            
            # Update form configuration metrics
            await self._update_form_metrics(session, performance_analysis)
            
            # Generate insights
            insights = await self._generate_form_insights(session, performance_analysis)
            
            return {
                "session_id": session_id,
                "completed": True,
                "session_duration": session_duration,
                "performance_analysis": performance_analysis,
                "insights": insights,
                "optimization_recommendations": await self._generate_optimization_recommendations(session)
            }
            
        except Exception as e:
            logger.error(f"Error completing form session: {e}")
            raise
    
    async def get_form_analytics_dashboard(self, form_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive form analytics dashboard"""
        
        try:
            # This would typically query a database
            # For now, return sample analytics
            
            return {
                "summary": {
                    "total_form_sessions": 8420,
                    "completion_rate": 0.342,
                    "avg_completion_time": 185,  # seconds
                    "mobile_completion_rate": 0.298,
                    "desktop_completion_rate": 0.378
                },
                "form_performance": {
                    "insurance_quote": {
                        "sessions": 5200,
                        "completion_rate": 0.356,
                        "avg_time": 165,
                        "top_abandonment_field": "phone"
                    }
                },
                "field_analytics": [
                    {"field_id": "first_name", "completion_rate": 0.95, "avg_time": 8, "error_rate": 0.02},
                    {"field_id": "email", "completion_rate": 0.89, "avg_time": 12, "error_rate": 0.08},
                    {"field_id": "phone", "completion_rate": 0.67, "avg_time": 18, "error_rate": 0.15},
                    {"field_id": "insurance_type", "completion_rate": 0.78, "avg_time": 15, "error_rate": 0.05}
                ],
                "optimization_impact": {
                    "mobile_optimization": {"improvement": 0.23, "sessions_affected": 3200},
                    "progressive_profiling": {"improvement": 0.18, "sessions_affected": 1800},
                    "field_reduction": {"improvement": 0.31, "sessions_affected": 2400}
                },
                "ab_test_results": [
                    {
                        "test_name": "Single vs Multi-step",
                        "variant_a": {"completion_rate": 0.32, "sessions": 2100},
                        "variant_b": {"completion_rate": 0.38, "sessions": 2050},
                        "winner": "variant_b",
                        "confidence": 0.95
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating form analytics dashboard: {e}")
            raise

# Global smart form optimization engine
smart_form_engine = SmartFormOptimizationEngine()