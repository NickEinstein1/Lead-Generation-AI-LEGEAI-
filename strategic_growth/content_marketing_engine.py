"""
Content Marketing Lead Generation Engine

Comprehensive content marketing system for blog posts, webinars, whitepapers,
ebooks, and other content-driven lead capture mechanisms.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ContentType(Enum):
    BLOG_POST = "blog_post"
    WHITEPAPER = "whitepaper"
    EBOOK = "ebook"
    WEBINAR = "webinar"
    CASE_STUDY = "case_study"
    INFOGRAPHIC = "infographic"
    VIDEO = "video"
    PODCAST = "podcast"
    NEWSLETTER = "newsletter"
    CHECKLIST = "checklist"

class ContentStage(Enum):
    AWARENESS = "awareness"
    CONSIDERATION = "consideration"
    DECISION = "decision"
    RETENTION = "retention"

class LeadMagnetType(Enum):
    GATED_CONTENT = "gated_content"
    EMAIL_SIGNUP = "email_signup"
    WEBINAR_REGISTRATION = "webinar_registration"
    FREE_CONSULTATION = "free_consultation"
    CALCULATOR_TOOL = "calculator_tool"
    ASSESSMENT = "assessment"

@dataclass
class ContentAsset:
    """Content marketing asset"""
    asset_id: str
    title: str
    content_type: ContentType
    stage: ContentStage
    
    # Content Details
    description: str
    author: str
    publish_date: datetime
    url: str
    
    # SEO and Marketing
    keywords: List[str] = field(default_factory=list)
    meta_description: str = ""
    featured_image: Optional[str] = None
    
    # Lead Generation
    lead_magnet_type: Optional[LeadMagnetType] = None
    cta_text: str = ""
    landing_page_url: Optional[str] = None
    
    # Performance Tracking
    views: int = 0
    downloads: int = 0
    leads_generated: int = 0
    conversion_rate: float = 0.0
    
    # Content Metadata
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    target_audience: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ContentLead:
    """Lead generated from content marketing"""
    lead_id: str
    content_asset_id: str
    content_type: ContentType
    lead_magnet_type: LeadMagnetType
    
    # Lead Information
    name: str
    email: str
    phone: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    
    # Engagement Data
    source_url: str
    referrer: Optional[str] = None
    utm_parameters: Dict[str, str] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)
    
    # Content Interaction
    time_on_page: Optional[int] = None  # seconds
    pages_viewed: int = 1
    content_consumed: List[str] = field(default_factory=list)
    
    # Lead Scoring Context
    interest_level: str = "medium"
    engagement_score: float = 0.0
    content_stage: ContentStage = ContentStage.AWARENESS
    
    # Metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_type: Optional[str] = None
    location: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WebinarEvent:
    """Webinar event details"""
    webinar_id: str
    title: str
    description: str
    presenter: str
    
    # Scheduling
    scheduled_date: datetime
    duration_minutes: int
    timezone: str
    
    # Registration
    registration_url: str
    max_attendees: Optional[int] = None
    registration_deadline: Optional[datetime] = None
    
    # Content
    agenda: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    target_audience: List[str] = field(default_factory=list)
    
    # Performance
    registrations: int = 0
    attendees: int = 0
    completion_rate: float = 0.0
    leads_generated: int = 0
    
    # Follow-up
    recording_url: Optional[str] = None
    follow_up_sequence: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)

class ContentMarketingEngine:
    """Main content marketing lead generation engine"""
    
    def __init__(self):
        self.content_assets = {}
        self.webinar_events = {}
        self.lead_magnets = {}
        self.performance_analytics = ContentAnalytics()
    
    async def create_content_asset(self, asset_data: Dict[str, Any]) -> ContentAsset:
        """Create a new content marketing asset"""
        
        try:
            asset = ContentAsset(
                asset_id=asset_data['asset_id'],
                title=asset_data['title'],
                content_type=ContentType(asset_data['content_type']),
                stage=ContentStage(asset_data['stage']),
                description=asset_data['description'],
                author=asset_data['author'],
                publish_date=asset_data.get('publish_date', datetime.now()),
                url=asset_data['url'],
                keywords=asset_data.get('keywords', []),
                meta_description=asset_data.get('meta_description', ''),
                lead_magnet_type=LeadMagnetType(asset_data['lead_magnet_type']) if asset_data.get('lead_magnet_type') else None,
                cta_text=asset_data.get('cta_text', ''),
                landing_page_url=asset_data.get('landing_page_url'),
                tags=asset_data.get('tags', []),
                categories=asset_data.get('categories', []),
                target_audience=asset_data.get('target_audience', [])
            )
            
            self.content_assets[asset.asset_id] = asset
            
            # Set up tracking and analytics
            await self._setup_content_tracking(asset)
            
            logger.info(f"Created content asset: {asset.title}")
            return asset
            
        except Exception as e:
            logger.error(f"Error creating content asset: {e}")
            raise
    
    async def capture_content_lead(self, lead_data: Dict[str, Any]) -> ContentLead:
        """Capture a lead from content marketing"""
        
        try:
            # Validate required fields
            required_fields = ['email', 'content_asset_id', 'source_url']
            for field in required_fields:
                if field not in lead_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Get content asset
            asset_id = lead_data['content_asset_id']
            if asset_id not in self.content_assets:
                raise ValueError(f"Content asset not found: {asset_id}")
            
            asset = self.content_assets[asset_id]
            
            # Create lead
            lead = ContentLead(
                lead_id=f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{lead_data['email'].split('@')[0]}",
                content_asset_id=asset_id,
                content_type=asset.content_type,
                lead_magnet_type=asset.lead_magnet_type or LeadMagnetType.EMAIL_SIGNUP,
                name=lead_data.get('name', ''),
                email=lead_data['email'],
                phone=lead_data.get('phone'),
                company=lead_data.get('company'),
                job_title=lead_data.get('job_title'),
                source_url=lead_data['source_url'],
                referrer=lead_data.get('referrer'),
                utm_parameters=lead_data.get('utm_parameters', {}),
                session_data=lead_data.get('session_data', {}),
                time_on_page=lead_data.get('time_on_page'),
                pages_viewed=lead_data.get('pages_viewed', 1),
                content_stage=asset.stage,
                ip_address=lead_data.get('ip_address'),
                user_agent=lead_data.get('user_agent'),
                device_type=lead_data.get('device_type'),
                location=lead_data.get('location'),
                raw_data=lead_data
            )
            
            # Calculate engagement score
            lead.engagement_score = await self._calculate_content_engagement_score(lead, asset)
            
            # Determine interest level
            lead.interest_level = await self._determine_interest_level(lead, asset)
            
            # Update asset performance
            asset.leads_generated += 1
            asset.conversion_rate = asset.leads_generated / max(asset.views, 1)
            
            # Trigger follow-up sequences
            await self._trigger_content_follow_up(lead, asset)
            
            logger.info(f"Captured content lead: {lead.email} from {asset.title}")
            return lead
            
        except Exception as e:
            logger.error(f"Error capturing content lead: {e}")
            raise
    
    async def create_webinar_event(self, webinar_data: Dict[str, Any]) -> WebinarEvent:
        """Create a new webinar event"""
        
        try:
            webinar = WebinarEvent(
                webinar_id=webinar_data['webinar_id'],
                title=webinar_data['title'],
                description=webinar_data['description'],
                presenter=webinar_data['presenter'],
                scheduled_date=webinar_data['scheduled_date'],
                duration_minutes=webinar_data['duration_minutes'],
                timezone=webinar_data.get('timezone', 'UTC'),
                registration_url=webinar_data['registration_url'],
                max_attendees=webinar_data.get('max_attendees'),
                registration_deadline=webinar_data.get('registration_deadline'),
                agenda=webinar_data.get('agenda', []),
                learning_objectives=webinar_data.get('learning_objectives', []),
                target_audience=webinar_data.get('target_audience', []),
                follow_up_sequence=webinar_data.get('follow_up_sequence', [])
            )
            
            self.webinar_events[webinar.webinar_id] = webinar
            
            # Set up webinar tracking
            await self._setup_webinar_tracking(webinar)
            
            logger.info(f"Created webinar event: {webinar.title}")
            return webinar
            
        except Exception as e:
            logger.error(f"Error creating webinar event: {e}")
            raise
    
    async def register_webinar_attendee(self, webinar_id: str, attendee_data: Dict[str, Any]) -> ContentLead:
        """Register an attendee for a webinar"""
        
        try:
            if webinar_id not in self.webinar_events:
                raise ValueError(f"Webinar not found: {webinar_id}")
            
            webinar = self.webinar_events[webinar_id]
            
            # Create lead from webinar registration
            lead_data = {
                'content_asset_id': f"webinar_{webinar_id}",
                'email': attendee_data['email'],
                'name': attendee_data.get('name', ''),
                'phone': attendee_data.get('phone'),
                'company': attendee_data.get('company'),
                'job_title': attendee_data.get('job_title'),
                'source_url': webinar.registration_url,
                'utm_parameters': attendee_data.get('utm_parameters', {}),
                'session_data': attendee_data.get('session_data', {})
            }
            
            # Create content asset for webinar if not exists
            if f"webinar_{webinar_id}" not in self.content_assets:
                await self._create_webinar_content_asset(webinar)
            
            lead = await self.capture_content_lead(lead_data)
            
            # Update webinar registration count
            webinar.registrations += 1
            
            # Send confirmation email
            await self._send_webinar_confirmation(lead, webinar)
            
            logger.info(f"Registered webinar attendee: {lead.email} for {webinar.title}")
            return lead
            
        except Exception as e:
            logger.error(f"Error registering webinar attendee: {e}")
            raise
    
    async def track_content_engagement(self, asset_id: str, engagement_data: Dict[str, Any]):
        """Track content engagement metrics"""
        
        try:
            if asset_id not in self.content_assets:
                logger.warning(f"Content asset not found for tracking: {asset_id}")
                return
            
            asset = self.content_assets[asset_id]
            
            # Update view count
            if engagement_data.get('event_type') == 'page_view':
                asset.views += 1
            
            # Update download count
            elif engagement_data.get('event_type') == 'download':
                asset.downloads += 1
            
            # Update conversion rate
            asset.conversion_rate = asset.leads_generated / max(asset.views, 1)
            
            # Store detailed engagement data
            await self.performance_analytics.track_engagement(asset_id, engagement_data)
            
        except Exception as e:
            logger.error(f"Error tracking content engagement: {e}")
    
    async def get_content_performance(self, asset_id: str = None, 
                                   date_range: Dict[str, datetime] = None) -> Dict[str, Any]:
        """Get content performance analytics"""
        
        try:
            if asset_id:
                # Single asset performance
                if asset_id not in self.content_assets:
                    raise ValueError(f"Content asset not found: {asset_id}")
                
                asset = self.content_assets[asset_id]
                return await self._get_asset_performance(asset, date_range)
            else:
                # Overall content performance
                return await self._get_overall_content_performance(date_range)
                
        except Exception as e:
            logger.error(f"Error getting content performance: {e}")
            return {}
    
    async def _calculate_content_engagement_score(self, lead: ContentLead, asset: ContentAsset) -> float:
        """Calculate engagement score for content lead"""
        
        score = 0.0
        
        # Base score for lead capture
        score += 20.0
        
        # Time on page bonus
        if lead.time_on_page:
            if lead.time_on_page > 300:  # 5+ minutes
                score += 15.0
            elif lead.time_on_page > 120:  # 2+ minutes
                score += 10.0
            elif lead.time_on_page > 60:  # 1+ minute
                score += 5.0
        
        # Content type bonus
        content_scores = {
            ContentType.WHITEPAPER: 15.0,
            ContentType.EBOOK: 15.0,
            ContentType.WEBINAR: 20.0,
            ContentType.CASE_STUDY: 12.0,
            ContentType.BLOG_POST: 5.0
        }
        score += content_scores.get(asset.content_type, 5.0)
        
        # Stage bonus (decision stage content is more valuable)
        stage_scores = {
            ContentStage.DECISION: 15.0,
            ContentStage.CONSIDERATION: 10.0,
            ContentStage.AWARENESS: 5.0,
            ContentStage.RETENTION: 8.0
        }
        score += stage_scores.get(asset.stage, 5.0)
        
        # UTM source bonus
        if lead.utm_parameters.get('utm_source') in ['google', 'linkedin', 'email']:
            score += 5.0
        
        # Company information bonus
        if lead.company:
            score += 10.0
        if lead.job_title:
            score += 5.0
        
        return min(score, 100.0)  # Cap at 100
    
    async def _determine_interest_level(self, lead: ContentLead, asset: ContentAsset) -> str:
        """Determine lead interest level based on content interaction"""
        
        if lead.engagement_score >= 70:
            return "high"
        elif lead.engagement_score >= 50:
            return "medium"
        else:
            return "low"
    
    async def _setup_content_tracking(self, asset: ContentAsset):
        """Set up tracking for content asset"""
        # Implementation would set up analytics tracking
        pass
    
    async def _setup_webinar_tracking(self, webinar: WebinarEvent):
        """Set up tracking for webinar event"""
        # Implementation would set up webinar platform integration
        pass
    
    async def _create_webinar_content_asset(self, webinar: WebinarEvent):
        """Create content asset for webinar"""
        
        asset_data = {
            'asset_id': f"webinar_{webinar.webinar_id}",
            'title': webinar.title,
            'content_type': 'webinar',
            'stage': 'consideration',
            'description': webinar.description,
            'author': webinar.presenter,
            'publish_date': webinar.scheduled_date,
            'url': webinar.registration_url,
            'lead_magnet_type': 'webinar_registration',
            'cta_text': 'Register Now',
            'target_audience': webinar.target_audience
        }
        
        await self.create_content_asset(asset_data)
    
    async def _trigger_content_follow_up(self, lead: ContentLead, asset: ContentAsset):
        """Trigger follow-up sequences for content leads"""
        
        # Implementation would trigger email sequences, sales notifications, etc.
        logger.info(f"Triggering follow-up for {lead.email} - {asset.title}")
    
    async def _send_webinar_confirmation(self, lead: ContentLead, webinar: WebinarEvent):
        """Send webinar confirmation email"""
        
        # Implementation would send confirmation email with calendar invite
        logger.info(f"Sending webinar confirmation to {lead.email} for {webinar.title}")
    
    async def _get_asset_performance(self, asset: ContentAsset, date_range: Dict[str, datetime] = None) -> Dict[str, Any]:
        """Get performance metrics for a single asset"""
        
        return {
            "asset_id": asset.asset_id,
            "title": asset.title,
            "content_type": asset.content_type.value,
            "stage": asset.stage.value,
            "metrics": {
                "views": asset.views,
                "downloads": asset.downloads,
                "leads_generated": asset.leads_generated,
                "conversion_rate": asset.conversion_rate,
                "engagement_score": 7.5  # Would be calculated from actual data
            },
            "performance_trend": "increasing",  # Would be calculated
            "top_traffic_sources": [
                {"source": "organic_search", "percentage": 0.45},
                {"source": "social_media", "percentage": 0.25},
                {"source": "email", "percentage": 0.20},
                {"source": "direct", "percentage": 0.10}
            ]
        }
    
    async def _get_overall_content_performance(self, date_range: Dict[str, datetime] = None) -> Dict[str, Any]:
        """Get overall content marketing performance"""
        
        return {
            "summary": {
                "total_assets": len(self.content_assets),
                "total_views": sum(asset.views for asset in self.content_assets.values()),
                "total_leads": sum(asset.leads_generated for asset in self.content_assets.values()),
                "average_conversion_rate": 0.085,
                "top_performing_type": "webinar"
            },
            "content_type_performance": {
                "webinar": {"leads": 450, "conversion_rate": 0.15},
                "whitepaper": {"leads": 320, "conversion_rate": 0.12},
                "ebook": {"leads": 280, "conversion_rate": 0.10},
                "blog_post": {"leads": 180, "conversion_rate": 0.03}
            },
            "stage_performance": {
                "awareness": {"leads": 520, "conversion_rate": 0.05},
                "consideration": {"leads": 480, "conversion_rate": 0.12},
                "decision": {"leads": 230, "conversion_rate": 0.18}
            }
        }

class ContentAnalytics:
    """Analytics engine for content marketing performance"""
    
    def __init__(self):
        self.engagement_data = {}
    
    async def track_engagement(self, asset_id: str, engagement_data: Dict[str, Any]):
        """Track detailed engagement metrics"""
        
        if asset_id not in self.engagement_data:
            self.engagement_data[asset_id] = []
        
        engagement_data['timestamp'] = datetime.now()
        self.engagement_data[asset_id].append(engagement_data)
    
    async def get_engagement_analytics(self, asset_id: str) -> Dict[str, Any]:
        """Get detailed engagement analytics for an asset"""
        
        if asset_id not in self.engagement_data:
            return {}
        
        data = self.engagement_data[asset_id]
        
        return {
            "total_interactions": len(data),
            "unique_visitors": len(set(item.get('visitor_id') for item in data if item.get('visitor_id'))),
            "average_time_on_page": sum(item.get('time_on_page', 0) for item in data) / len(data),
            "bounce_rate": 0.25,  # Would be calculated from actual data
            "scroll_depth": {
                "25%": 0.85,
                "50%": 0.65,
                "75%": 0.45,
                "100%": 0.25
            }
        }

# Global content marketing engine
content_marketing_engine = ContentMarketingEngine()