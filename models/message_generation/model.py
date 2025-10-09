import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import random
from datetime import datetime, timedelta

class MessageType(Enum):
    EMAIL = "email"
    SMS = "sms"
    CALL_SCRIPT = "call_script"
    SOCIAL_MEDIA = "social_media"
    FOLLOW_UP = "follow_up"

class MessageTone(Enum):
    URGENT = "urgent"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    EDUCATIONAL = "educational"
    CONSULTATIVE = "consultative"

@dataclass
class GeneratedMessage:
    message_type: str
    subject_line: str
    content: str
    call_to_action: str
    tone: str
    urgency_level: str
    personalization_elements: List[str]
    follow_up_timing: str
    compliance_notes: str

class MessageGenerationModel:
    """
    AI-powered message generation based on lead scoring results
    """
    
    def __init__(self):
        self.message_templates = self._load_message_templates()
        self.personalization_rules = self._load_personalization_rules()
        self.compliance_guidelines = self._load_compliance_guidelines()
        
    def generate_message(self, lead_data: Dict[str, Any], 
                        scoring_result: Dict[str, Any],
                        message_type: MessageType = MessageType.EMAIL) -> GeneratedMessage:
        """
        Generate personalized message based on lead scoring
        """
        # Determine message strategy
        strategy = self._determine_message_strategy(scoring_result)
        
        # Select appropriate tone
        tone = self._select_message_tone(scoring_result, strategy)
        
        # Generate personalized content
        content = self._generate_personalized_content(
            lead_data, scoring_result, message_type, tone, strategy
        )
        
        # Create subject line
        subject_line = self._generate_subject_line(
            lead_data, scoring_result, strategy
        )
        
        # Generate call to action
        cta = self._generate_call_to_action(scoring_result, strategy)
        
        # Determine follow-up timing
        follow_up_timing = self._determine_follow_up_timing(scoring_result)
        
        # Add compliance notes
        compliance_notes = self._generate_compliance_notes(
            lead_data, scoring_result, message_type
        )
        
        # Extract personalization elements
        personalization_elements = self._extract_personalization_elements(
            lead_data, scoring_result
        )
        
        return GeneratedMessage(
            message_type=message_type.value,
            subject_line=subject_line,
            content=content,
            call_to_action=cta,
            tone=tone.value,
            urgency_level=scoring_result.get('priority_level', 'MEDIUM'),
            personalization_elements=personalization_elements,
            follow_up_timing=follow_up_timing,
            compliance_notes=compliance_notes
        )
    
    def _determine_message_strategy(self, scoring_result: Dict[str, Any]) -> str:
        """Determine the overall messaging strategy"""
        priority = scoring_result.get('priority_level', 'MEDIUM')
        urgency_signals = scoring_result.get('urgency_signals', [])
        recommended_products = scoring_result.get('recommended_products', [])
        
        # Critical priority with urgency signals
        if priority == 'CRITICAL' and urgency_signals:
            return "URGENT_ACTION_REQUIRED"
        
        # High priority with multiple products
        elif priority == 'HIGH' and len(recommended_products) > 1:
            return "COMPREHENSIVE_SOLUTION"
        
        # High engagement leads
        elif scoring_result.get('conversion_velocity', {}).get(recommended_products[0] if recommended_products else 'base') == 'IMMEDIATE':
            return "FAST_TRACK_CONVERSION"
        
        # Educational approach for slower leads
        elif scoring_result.get('overall_score', 0) < 60:
            return "EDUCATIONAL_NURTURE"
        
        # Standard consultative approach
        else:
            return "CONSULTATIVE_APPROACH"
    
    def _select_message_tone(self, scoring_result: Dict[str, Any], strategy: str) -> MessageTone:
        """Select appropriate message tone"""
        urgency_signals = scoring_result.get('urgency_signals', [])
        
        if strategy == "URGENT_ACTION_REQUIRED":
            return MessageTone.URGENT
        elif strategy == "EDUCATIONAL_NURTURE":
            return MessageTone.EDUCATIONAL
        elif strategy == "COMPREHENSIVE_SOLUTION":
            return MessageTone.CONSULTATIVE
        elif urgency_signals:
            return MessageTone.PROFESSIONAL
        else:
            return MessageTone.FRIENDLY
    
    def _generate_personalized_content(self, lead_data: Dict[str, Any],
                                     scoring_result: Dict[str, Any],
                                     message_type: MessageType,
                                     tone: MessageTone,
                                     strategy: str) -> str:
        """Generate the main message content"""
        
        # Get lead details
        name = lead_data.get('name', 'there')
        age = lead_data.get('age', 35)
        income = lead_data.get('income', 50000)
        family_size = lead_data.get('family_size', 1)
        location = lead_data.get('location', 'your area')
        
        # Get scoring insights
        primary_product = scoring_result.get('recommended_products', ['insurance'])[0]
        urgency_signals = scoring_result.get('urgency_signals', [])
        revenue_potential = scoring_result.get('revenue_potential', 0)
        
        # Base greeting
        greeting = self._generate_greeting(name, tone)
        
        # Opening hook based on strategy
        opening = self._generate_opening_hook(strategy, urgency_signals, primary_product)
        
        # Value proposition
        value_prop = self._generate_value_proposition(
            lead_data, scoring_result, primary_product
        )
        
        # Social proof
        social_proof = self._generate_social_proof(lead_data, primary_product)
        
        # Urgency element
        urgency_element = self._generate_urgency_element(urgency_signals, strategy)
        
        # Closing
        closing = self._generate_closing(tone, strategy)
        
        # Combine all elements
        if message_type == MessageType.EMAIL:
            content = f"""{greeting}

{opening}

{value_prop}

{social_proof}

{urgency_element}

{closing}

Best regards,
[Agent Name]
[Company Name]
[Phone] | [Email]"""
        
        elif message_type == MessageType.SMS:
            content = f"{greeting} {opening} {urgency_element} Reply YES for free quote or call [PHONE]."
        
        elif message_type == MessageType.CALL_SCRIPT:
            content = f"""OPENING: {greeting} {opening}

VALUE PROPOSITION: {value_prop}

HANDLE OBJECTIONS: [Reference objection handling from scoring]

URGENCY: {urgency_element}

CLOSE: {closing}

NEXT STEPS: [Schedule follow-up based on velocity prediction]"""
        
        return content
    
    def _generate_greeting(self, name: str, tone: MessageTone) -> str:
        """Generate appropriate greeting"""
        if tone == MessageTone.URGENT:
            return f"Hi {name},"
        elif tone == MessageTone.PROFESSIONAL:
            return f"Dear {name},"
        elif tone == MessageTone.FRIENDLY:
            return f"Hello {name}!"
        elif tone == MessageTone.EDUCATIONAL:
            return f"Hi {name},"
        else:
            return f"Good day {name},"
    
    def _generate_opening_hook(self, strategy: str, urgency_signals: List[str], 
                             primary_product: str) -> str:
        """Generate compelling opening hook"""
        hooks = {
            "URGENT_ACTION_REQUIRED": [
                f"I noticed you're in a time-sensitive situation regarding {primary_product} insurance.",
                f"There's an important deadline approaching for your {primary_product} coverage.",
                f"I wanted to reach out immediately about your {primary_product} insurance needs."
            ],
            "COMPREHENSIVE_SOLUTION": [
                f"I've identified several insurance opportunities that could save you money.",
                f"Based on your profile, I can help you with a complete insurance solution.",
                f"I have some exciting news about bundling your insurance needs."
            ],
            "FAST_TRACK_CONVERSION": [
                f"You seem ready to move forward with {primary_product} insurance.",
                f"I can get you a competitive {primary_product} quote within 24 hours.",
                f"Let's fast-track your {primary_product} insurance application."
            ],
            "EDUCATIONAL_NURTURE": [
                f"I wanted to share some important information about {primary_product} insurance.",
                f"Many people in your situation benefit from understanding {primary_product} options.",
                f"I'd like to help you explore {primary_product} insurance at your own pace."
            ],
            "CONSULTATIVE_APPROACH": [
                f"I'd love to discuss how {primary_product} insurance fits into your financial plan.",
                f"Let's explore the best {primary_product} options for your unique situation.",
                f"I'm here to help you make an informed decision about {primary_product} insurance."
            ]
        }
        
        return random.choice(hooks.get(strategy, hooks["CONSULTATIVE_APPROACH"]))
    
    def _generate_value_proposition(self, lead_data: Dict[str, Any],
                                  scoring_result: Dict[str, Any],
                                  primary_product: str) -> str:
        """Generate personalized value proposition"""
        age = lead_data.get('age', 35)
        income = lead_data.get('income', 50000)
        family_size = lead_data.get('family_size', 1)
        urgency_signals = scoring_result.get('urgency_signals', [])
        
        value_props = []
        
        # Age-based value props
        if primary_product == 'life' and age < 40:
            value_props.append("At your age, life insurance rates are at their lowest.")
        elif primary_product == 'healthcare' and age > 50:
            value_props.append("Healthcare costs increase significantly after 50 - let's protect you.")
        
        # Family-based value props
        if family_size > 1:
            value_props.append(f"With {family_size} family members, comprehensive coverage is essential.")
        
        # Income-based value props
        if income > 75000:
            value_props.append("Your income level qualifies you for premium coverage options.")
        
        # Urgency-based value props
        if "NEW_BABY" in urgency_signals:
            value_props.append("Congratulations! New parents often increase coverage by 10x.")
        elif "NO_CURRENT_COVERAGE" in urgency_signals:
            value_props.append("Being uninsured puts your family at significant financial risk.")
        
        # Default value prop
        if not value_props:
            value_props.append(f"Our {primary_product} insurance provides peace of mind and financial security.")
        
        return " ".join(value_props[:2])  # Use top 2 value props
    
    def _generate_social_proof(self, lead_data: Dict[str, Any], primary_product: str) -> str:
        """Generate relevant social proof"""
        location = lead_data.get('location', 'your area')
        age = lead_data.get('age', 35)
        
        social_proofs = [
            f"Over 10,000 families in {location} trust us with their {primary_product} insurance.",
            f"95% of our clients in the {age-5}-{age+5} age group recommend our {primary_product} plans.",
            f"We've helped save families in {location} an average of $1,200 annually on {primary_product} insurance.",
            f"Rated #1 {primary_product} insurance provider in {location} for customer satisfaction."
        ]
        
        return random.choice(social_proofs)
    
    def _generate_urgency_element(self, urgency_signals: List[str], strategy: str) -> str:
        """Generate urgency-based messaging"""
        if not urgency_signals and strategy != "URGENT_ACTION_REQUIRED":
            return "The best time to get insurance is when you don't need it yet."
        
        urgency_messages = {
            "OPEN_ENROLLMENT_PERIOD": "Open enrollment ends soon - don't miss this opportunity!",
            "NEW_BABY": "New parents should update coverage within 30 days for optimal rates.",
            "NO_CURRENT_COVERAGE": "Every day without coverage puts your family at risk.",
            "JOB_CHANGE": "Job transitions are the perfect time to review and upgrade coverage.",
            "INCOME_INCREASE": "Your increased income qualifies you for better coverage options.",
            "AGE_URGENCY": "Insurance rates increase with age - lock in today's rates now."
        }
        
        if urgency_signals:
            return urgency_messages.get(urgency_signals[0], "Time-sensitive opportunity - let's discuss today.")
        
        return "I'd love to help you explore your options at your convenience."
    
    def _generate_closing(self, tone: MessageTone, strategy: str) -> str:
        """Generate appropriate closing"""
        closings = {
            MessageTone.URGENT: "I'm standing by to help you secure coverage immediately.",
            MessageTone.PROFESSIONAL: "I look forward to discussing your insurance needs.",
            MessageTone.FRIENDLY: "I'm excited to help you find the perfect coverage!",
            MessageTone.EDUCATIONAL: "I'm here to answer any questions you might have.",
            MessageTone.CONSULTATIVE: "Let's schedule a time to review your options together."
        }
        
        return closings.get(tone, "I'm here to help with all your insurance needs.")
    
    def _generate_subject_line(self, lead_data: Dict[str, Any],
                             scoring_result: Dict[str, Any],
                             strategy: str) -> str:
        """Generate compelling subject line"""
        name = lead_data.get('name', 'there')
        primary_product = scoring_result.get('recommended_products', ['insurance'])[0]
        urgency_signals = scoring_result.get('urgency_signals', [])
        
        subject_templates = {
            "URGENT_ACTION_REQUIRED": [
                f"⏰ {name}, urgent: {primary_product} deadline approaching",
                f"Time-sensitive {primary_product} opportunity for {name}",
                f"Action required: {name}'s {primary_product} coverage"
            ],
            "COMPREHENSIVE_SOLUTION": [
                f"{name}, save $1,200+ with bundled insurance",
                f"Complete insurance solution for {name}",
                f"{name}, your personalized insurance package is ready"
            ],
            "FAST_TRACK_CONVERSION": [
                f"{name}, your {primary_product} quote in 24 hours",
                f"Fast-track {primary_product} approval for {name}",
                f"{name}, let's finalize your {primary_product} today"
            ],
            "EDUCATIONAL_NURTURE": [
                f"{name}, important {primary_product} information inside",
                f"Insurance guide for {name}",
                f"{name}, understanding your {primary_product} options"
            ],
            "CONSULTATIVE_APPROACH": [
                f"{name}, let's discuss your {primary_product} needs",
                f"Personalized {primary_product} consultation for {name}",
                f"{name}, your insurance questions answered"
            ]
        }
        
        return random.choice(subject_templates.get(strategy, subject_templates["CONSULTATIVE_APPROACH"]))
    
    def _generate_call_to_action(self, scoring_result: Dict[str, Any], strategy: str) -> str:
        """Generate appropriate call to action"""
        conversion_velocity = scoring_result.get('conversion_velocity', {})
        primary_product = scoring_result.get('recommended_products', ['insurance'])[0]
        
        # High velocity leads
        if 'IMMEDIATE' in conversion_velocity.values():
            return "Call me now at [PHONE] or reply to schedule your consultation today!"
        
        # Fast velocity leads
        elif 'FAST' in conversion_velocity.values():
            return "Reply with your best time for a 15-minute call, or click here for an instant quote."
        
        # Strategy-based CTAs
        cta_map = {
            "URGENT_ACTION_REQUIRED": "Call immediately: [PHONE] - don't wait!",
            "COMPREHENSIVE_SOLUTION": "Schedule your free consultation: [CALENDAR_LINK]",
            "FAST_TRACK_CONVERSION": "Get your quote now: [QUOTE_LINK]",
            "EDUCATIONAL_NURTURE": "Download our free guide: [GUIDE_LINK]",
            "CONSULTATIVE_APPROACH": "Book a no-obligation consultation: [CALENDAR_LINK]"
        }
        
        return cta_map.get(strategy, "Contact me to learn more: [PHONE] or [EMAIL]")
    
    def _determine_follow_up_timing(self, scoring_result: Dict[str, Any]) -> str:
        """Determine when to follow up"""
        optimal_timing = scoring_result.get('optimal_contact_time', 'WITHIN_24_HOURS')
        priority = scoring_result.get('priority_level', 'MEDIUM')
        
        timing_map = {
            'IMMEDIATE': 'Follow up in 2 hours if no response',
            'WITHIN_2_HOURS': 'Follow up in 4 hours if no response',
            'WITHIN_4_HOURS': 'Follow up in 8 hours if no response',
            'WITHIN_24_HOURS': 'Follow up in 2 days if no response',
            'WITHIN_3_DAYS': 'Follow up in 1 week if no response'
        }
        
        return timing_map.get(optimal_timing, 'Follow up in 3 days if no response')
    
    def _generate_compliance_notes(self, lead_data: Dict[str, Any],
                                 scoring_result: Dict[str, Any],
                                 message_type: MessageType) -> str:
        """Generate compliance reminders"""
        notes = []
        
        # Consent verification
        if not lead_data.get('consent_given', False):
            notes.append("⚠️ Verify consent before sending")
        
        # TCPA compliance for SMS
        if message_type == MessageType.SMS:
            notes.append("📱 TCPA: Include opt-out instructions")
        
        # CAN-SPAM for email
        if message_type == MessageType.EMAIL:
            notes.append("📧 CAN-SPAM: Include unsubscribe link")
        
        # State-specific regulations
        location = lead_data.get('location', '')
        if 'CA' in location or 'California' in location:
            notes.append("🏛️ CA: Include privacy notice")
        
        # Age-related compliance
        age = lead_data.get('age', 35)
        if age < 18:
            notes.append("🔞 Minor: Parental consent required")
        
        return " | ".join(notes) if notes else "✅ Standard compliance applies"
    
    def _extract_personalization_elements(self, lead_data: Dict[str, Any],
                                        scoring_result: Dict[str, Any]) -> List[str]:
        """Extract elements used for personalization"""
        elements = []
        
        if lead_data.get('name'):
            elements.append("Name")
        if lead_data.get('age'):
            elements.append("Age")
        if lead_data.get('location'):
            elements.append("Location")
        if lead_data.get('family_size', 1) > 1:
            elements.append("Family size")
        if scoring_result.get('urgency_signals'):
            elements.append("Urgency signals")
        if scoring_result.get('recommended_products'):
            elements.append("Product recommendations")
        
        return elements
    
    def _load_message_templates(self) -> Dict:
        """Load message templates (would be from database/files)"""
        return {
            "email_templates": {},
            "sms_templates": {},
            "call_scripts": {},
            "social_media": {}
        }
    
    def _load_personalization_rules(self) -> Dict:
        """Load personalization rules"""
        return {
            "age_based": {},
            "income_based": {},
            "location_based": {},
            "urgency_based": {}
        }
    
    def _load_compliance_guidelines(self) -> Dict:
        """Load compliance guidelines"""
        return {
            "tcpa": {},
            "can_spam": {},
            "state_regulations": {},
            "industry_specific": {}
        }