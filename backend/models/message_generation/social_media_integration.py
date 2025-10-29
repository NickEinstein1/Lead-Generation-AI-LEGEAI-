from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import requests
from datetime import datetime

class SocialPlatform(Enum):
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    X = "x"  # Formerly Twitter
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"

class InteractionType(Enum):
    DIRECT_MESSAGE = "direct_message"
    COMMENT_REPLY = "comment_reply"
    POST_ENGAGEMENT = "post_engagement"
    STORY_REPLY = "story_reply"
    CONNECTION_REQUEST = "connection_request"

@dataclass
class SocialMediaMessage:
    platform: str
    interaction_type: str
    message_content: str
    character_limit: int
    includes_media: bool
    call_to_action: str
    hashtags: List[str]
    mentions: List[str]
    compliance_notes: str
    engagement_strategy: str

class SocialMediaMessageGenerator:
    """
    Generate interactive social media messages for lead engagement
    """
    
    def __init__(self):
        self.platform_limits = {
            "facebook": {"dm": 2000, "comment": 8000, "post": 63206},
            "instagram": {"dm": 1000, "comment": 2200, "story": 160},
            "linkedin": {"dm": 8000, "comment": 1250, "post": 3000},
            "x": {"dm": 10000, "post": 280, "reply": 280},  # Updated from twitter
            "tiktok": {"dm": 1000, "comment": 150},
            "youtube": {"comment": 10000}
        }
        
        self.platform_features = {
            "facebook": ["dm", "comment", "post", "story"],
            "instagram": ["dm", "comment", "story", "post"],
            "linkedin": ["dm", "comment", "post", "connection"],
            "x": ["dm", "reply", "mention", "repost"],  # Updated from twitter
            "tiktok": ["dm", "comment"],
            "youtube": ["comment", "community_post"]
        }
    
    def generate_social_message(self, lead_data: Dict[str, Any],
                              scoring_result: Dict[str, Any],
                              platform: SocialPlatform,
                              interaction_type: InteractionType,
                              context: Dict[str, Any] = None) -> SocialMediaMessage:
        """
        Generate platform-specific interactive message
        """
        
        # Get lead social profile data
        social_profile = lead_data.get('social_profiles', {}).get(platform.value, {})
        
        # Determine message strategy based on platform and lead score
        strategy = self._determine_social_strategy(scoring_result, platform, social_profile)
        
        # Generate platform-optimized content
        content = self._generate_platform_content(
            lead_data, scoring_result, platform, interaction_type, strategy, context
        )
        
        # Get character limit for platform/interaction type
        char_limit = self._get_character_limit(platform, interaction_type)
        
        # Optimize content for character limit
        optimized_content = self._optimize_for_character_limit(content, char_limit)
        
        # Generate hashtags and mentions
        hashtags = self._generate_hashtags(lead_data, scoring_result, platform)
        mentions = self._generate_mentions(lead_data, platform, context)
        
        # Create call to action
        cta = self._generate_social_cta(scoring_result, platform, interaction_type)
        
        # Determine media inclusion
        includes_media = self._should_include_media(platform, interaction_type, strategy)
        
        # Generate compliance notes
        compliance_notes = self._generate_social_compliance_notes(platform, interaction_type)
        
        # Determine engagement strategy
        engagement_strategy = self._determine_engagement_strategy(scoring_result, platform)
        
        return SocialMediaMessage(
            platform=platform.value,
            interaction_type=interaction_type.value,
            message_content=optimized_content,
            character_limit=char_limit,
            includes_media=includes_media,
            call_to_action=cta,
            hashtags=hashtags,
            mentions=mentions,
            compliance_notes=compliance_notes,
            engagement_strategy=engagement_strategy
        )
    
    def _determine_social_strategy(self, scoring_result: Dict[str, Any],
                                 platform: SocialPlatform,
                                 social_profile: Dict[str, Any]) -> str:
        """Determine social media engagement strategy"""
        
        priority = scoring_result.get('priority_level', 'MEDIUM')
        engagement_level = social_profile.get('engagement_level', 'LOW')
        follower_count = social_profile.get('followers', 0)
        
        # High-value leads with high engagement
        if priority in ['CRITICAL', 'HIGH'] and engagement_level == 'HIGH':
            return "DIRECT_PERSONAL_APPROACH"
        
        # Professional platforms
        elif platform == SocialPlatform.LINKEDIN:
            return "PROFESSIONAL_NETWORKING"
        
        # Visual platforms
        elif platform in [SocialPlatform.INSTAGRAM, SocialPlatform.TIKTOK]:
            return "VISUAL_STORYTELLING"
        
        # Influencers or high followers
        elif follower_count > 10000:
            return "INFLUENCER_COLLABORATION"
        
        # Standard approach
        else:
            return "VALUE_FIRST_ENGAGEMENT"
    
    def _generate_platform_content(self, lead_data: Dict[str, Any],
                                 scoring_result: Dict[str, Any],
                                 platform: SocialPlatform,
                                 interaction_type: InteractionType,
                                 strategy: str,
                                 context: Dict[str, Any]) -> str:
        """Generate platform-specific content"""
        
        name = lead_data.get('name', 'there')
        age = lead_data.get('age', 35)
        interests = lead_data.get('interests', [])
        primary_product = scoring_result.get('recommended_products', ['insurance'])[0]
        
        # Platform-specific content generation
        if platform == SocialPlatform.LINKEDIN:
            return self._generate_linkedin_content(name, strategy, primary_product, interaction_type)
        
        elif platform == SocialPlatform.INSTAGRAM:
            return self._generate_instagram_content(name, strategy, primary_product, interaction_type, interests)
        
        elif platform == SocialPlatform.FACEBOOK:
            return self._generate_facebook_content(name, strategy, primary_product, interaction_type, context)
        
        elif platform == SocialPlatform.X:  # Updated from TWITTER
            return self._generate_x_content(name, strategy, primary_product, interaction_type)
        
        elif platform == SocialPlatform.TIKTOK:
            return self._generate_tiktok_content(name, strategy, primary_product, interaction_type)
        
        else:
            return f"Hi {name}! I'd love to help you with {primary_product} insurance. Let's connect!"
    
    def _generate_linkedin_content(self, name: str, strategy: str, product: str, interaction_type: InteractionType) -> str:
        """Generate LinkedIn-specific content"""
        
        if interaction_type == InteractionType.CONNECTION_REQUEST:
            return f"Hi {name}, I help professionals like yourself with {product} insurance planning. Would love to connect and share some insights!"
        
        elif interaction_type == InteractionType.DIRECT_MESSAGE:
            if strategy == "PROFESSIONAL_NETWORKING":
                return f"""Hi {name},

I noticed your professional background and thought you might be interested in optimizing your {product} insurance strategy.

Many professionals in your field have saved 20-30% while getting better coverage.

Would you be open to a brief 15-minute conversation about your current situation?

Best regards,
[Your Name]"""
            
        elif interaction_type == InteractionType.COMMENT_REPLY:
            return f"Great point, {name}! This is exactly why having the right {product} insurance is so important for professionals. Happy to discuss strategies that work for your situation."
        
        return f"Hi {name}, I'd love to help you with professional {product} insurance planning."
    
    def _generate_instagram_content(self, name: str, strategy: str, product: str, 
                                  interaction_type: InteractionType, interests: List[str]) -> str:
        """Generate Instagram-specific content"""
        
        if interaction_type == InteractionType.DIRECT_MESSAGE:
            if strategy == "VISUAL_STORYTELLING":
                return f"Hey {name}! 👋 Love your content! I help people like you protect what matters most with smart {product} insurance. Mind if I share a quick tip that could save you money? 💰"
            
        elif interaction_type == InteractionType.COMMENT_REPLY:
            return f"@{name} This is so relatable! 😊 BTW, did you know you could be saving on {product} insurance? DM me for a free quote! 📱"
        
        elif interaction_type == InteractionType.STORY_REPLY:
            return f"Love this story! 🔥 Speaking of protecting what you love - have you reviewed your {product} insurance lately? I can help you save! 💪"
        
        return f"Hey {name}! 👋 Let's chat about {product} insurance!"
    
    def _generate_facebook_content(self, name: str, strategy: str, product: str,
                                 interaction_type: InteractionType, context: Dict[str, Any]) -> str:
        """Generate Facebook-specific content"""
        
        if interaction_type == InteractionType.DIRECT_MESSAGE:
            if strategy == "VALUE_FIRST_ENGAGEMENT":
                return f"""Hi {name}!

I hope you're doing well! I came across your profile and thought you might benefit from a free {product} insurance review.

Many families in your area have been able to:
✅ Save 25% on premiums
✅ Increase coverage by 50%
✅ Get better benefits

Would you like me to run a quick comparison for you? It's completely free and takes just 5 minutes.

Let me know if you're interested!

Best,
[Your Name]"""
        
        elif interaction_type == InteractionType.COMMENT_REPLY:
            post_topic = context.get('post_topic', 'life updates') if context else 'life updates'
            return f"@{name} Thanks for sharing! Life changes like this are perfect times to review {product} insurance. Happy to help if you need guidance! 😊"
        
        return f"Hi {name}! I'd love to help you with {product} insurance."
    
    def _generate_x_content(self, name: str, strategy: str, product: str, interaction_type: InteractionType) -> str:
        """Generate X (formerly Twitter) specific content"""
        
        if interaction_type == InteractionType.DIRECT_MESSAGE:
            return f"Hi {name}! Saw your posts about financial planning. I help people optimize {product} insurance - often saving 20-30%. Quick chat? 💬"
        
        elif interaction_type == InteractionType.COMMENT_REPLY:
            return f"@{name} Great post! This is why having proper {product} insurance is crucial. DM me for tips on optimizing coverage while saving money! 💰"
        
        return f"@{name} Let's talk {product} insurance! 📱"
    
    def _generate_tiktok_content(self, name: str, strategy: str, product: str, interaction_type: InteractionType) -> str:
        """Generate TikTok-specific content"""
        
        if interaction_type == InteractionType.DIRECT_MESSAGE:
            return f"Hey {name}! 🔥 Love your content! Quick question - are you paying too much for {product} insurance? I help creators save money! Want tips? 💰"
        
        elif interaction_type == InteractionType.COMMENT_REPLY:
            return f"@{name} This is fire! 🔥 BTW, creators like you need special {product} insurance. I can help you get covered for less! DM me! 📱"
        
        return f"Hey {name}! Let's talk {product} insurance for creators! 🎬"
    
    def _get_character_limit(self, platform: SocialPlatform, interaction_type: InteractionType) -> int:
        """Get character limit for platform and interaction type"""
        
        limits = {
            SocialPlatform.FACEBOOK: {
                InteractionType.DIRECT_MESSAGE: 2000,
                InteractionType.COMMENT_REPLY: 8000,
                InteractionType.POST_ENGAGEMENT: 63206
            },
            SocialPlatform.INSTAGRAM: {
                InteractionType.DIRECT_MESSAGE: 1000,
                InteractionType.COMMENT_REPLY: 2200,
                InteractionType.STORY_REPLY: 160
            },
            SocialPlatform.LINKEDIN: {
                InteractionType.DIRECT_MESSAGE: 8000,
                InteractionType.COMMENT_REPLY: 1250,
                InteractionType.CONNECTION_REQUEST: 300
            },
            SocialPlatform.X: {  # Updated from TWITTER
                InteractionType.DIRECT_MESSAGE: 10000,
                InteractionType.COMMENT_REPLY: 280
            },
            SocialPlatform.TIKTOK: {
                InteractionType.DIRECT_MESSAGE: 1000,
                InteractionType.COMMENT_REPLY: 150
            }
        }
        
        return limits.get(platform, {}).get(interaction_type, 1000)
    
    def _optimize_for_character_limit(self, content: str, char_limit: int) -> str:
        """Optimize content for character limit"""
        
        if len(content) <= char_limit:
            return content
        
        # Truncate and add continuation
        truncated = content[:char_limit-20]
        last_space = truncated.rfind(' ')
        
        if last_space > char_limit * 0.8:  # If we can find a good break point
            return truncated[:last_space] + "... (continued in DM)"
        else:
            return truncated + "..."
    
    def _generate_hashtags(self, lead_data: Dict[str, Any], scoring_result: Dict[str, Any], 
                         platform: SocialPlatform) -> List[str]:
        """Generate relevant hashtags"""
        
        primary_product = scoring_result.get('recommended_products', ['insurance'])[0]
        location = lead_data.get('location', '').replace(' ', '')
        age_group = "millennials" if lead_data.get('age', 35) < 40 else "genx"
        
        base_hashtags = [f"#{primary_product}insurance", "#financialplanning", "#protection"]
        
        # Platform-specific hashtags
        if platform == SocialPlatform.INSTAGRAM:
            base_hashtags.extend(["#instagood", "#lifestyle", f"#{age_group}"])
        elif platform == SocialPlatform.LINKEDIN:
            base_hashtags.extend(["#professional", "#career", "#finance"])
        elif platform == SocialPlatform.X:  # Updated from TWITTER
            base_hashtags.extend(["#fintech", "#insurtech", "#moneytips"])
        elif platform == SocialPlatform.TIKTOK:
            base_hashtags.extend(["#moneytips", "#adulting", "#financetok"])
        
        # Location-based
        if location:
            base_hashtags.append(f"#{location}")
        
        return base_hashtags[:5]  # Limit to 5 hashtags
    
    def _generate_mentions(self, lead_data: Dict[str, Any], platform: SocialPlatform, 
                         context: Dict[str, Any]) -> List[str]:
        """Generate relevant mentions"""
        
        mentions = []
        
        # If replying to a post, mention the original poster
        if context and context.get('original_poster'):
            mentions.append(f"@{context['original_poster']}")
        
        # Add lead's username if available
        social_profiles = lead_data.get('social_profiles', {})
        platform_profile = social_profiles.get(platform.value, {})
        username = platform_profile.get('username')
        
        if username:
            mentions.append(f"@{username}")
        
        return mentions
    
    def _generate_social_cta(self, scoring_result: Dict[str, Any], platform: SocialPlatform, 
                           interaction_type: InteractionType) -> str:
        """Generate platform-appropriate call to action"""
        
        priority = scoring_result.get('priority_level', 'MEDIUM')
        
        # High priority CTAs
        if priority in ['CRITICAL', 'HIGH']:
            cta_map = {
                SocialPlatform.LINKEDIN: "Send me a message to discuss your options",
                SocialPlatform.INSTAGRAM: "DM me for a free quote! 📱",
                SocialPlatform.FACEBOOK: "Message me for a free consultation",
                SocialPlatform.X: "DM for instant quote 💬",  # Updated from TWITTER
                SocialPlatform.TIKTOK: "DM me for money-saving tips! 💰"
            }
        else:
            cta_map = {
                SocialPlatform.LINKEDIN: "Connect with me to learn more",
                SocialPlatform.INSTAGRAM: "Follow for insurance tips! 💡",
                SocialPlatform.FACEBOOK: "Like this page for helpful tips",
                SocialPlatform.X: "Follow for financial tips 📈",  # Updated from TWITTER
                SocialPlatform.TIKTOK: "Follow for more money tips! 🔥"
            }
        
        return cta_map.get(platform, "Let's connect!")
    
    def _should_include_media(self, platform: SocialPlatform, interaction_type: InteractionType, 
                            strategy: str) -> bool:
        """Determine if message should include media"""
        
        # Visual platforms benefit from media
        visual_platforms = [SocialPlatform.INSTAGRAM, SocialPlatform.TIKTOK]
        
        # Direct messages typically don't include media initially
        if interaction_type == InteractionType.DIRECT_MESSAGE:
            return False
        
        # Visual storytelling strategy
        if strategy == "VISUAL_STORYTELLING" and platform in visual_platforms:
            return True
        
        # Professional platforms for infographics
        if platform == SocialPlatform.LINKEDIN and interaction_type == InteractionType.POST_ENGAGEMENT:
            return True
        
        return False
    
    def _generate_social_compliance_notes(self, platform: SocialPlatform, 
                                        interaction_type: InteractionType) -> str:
        """Generate compliance notes for social media"""
        
        notes = []
        
        # Platform-specific compliance
        if platform == SocialPlatform.LINKEDIN:
            notes.append("LinkedIn: Professional networking only, no spam")
        
        if interaction_type == InteractionType.DIRECT_MESSAGE:
            notes.append("DM: Ensure recipient has shown interest or engagement")
        
        # General social media compliance
        notes.extend([
            "Social: Respect platform terms of service",
            "Privacy: Don't share personal information publicly",
            "Consent: Only message users who have engaged with content"
        ])
        
        return " | ".join(notes)
    
    def _determine_engagement_strategy(self, scoring_result: Dict[str, Any], 
                                     platform: SocialPlatform) -> str:
        """Determine overall engagement strategy"""
        
        priority = scoring_result.get('priority_level', 'MEDIUM')
        
        if priority == 'CRITICAL':
            return "IMMEDIATE_DIRECT_OUTREACH"
        elif priority == 'HIGH':
            return "TARGETED_ENGAGEMENT"
        elif platform in [SocialPlatform.INSTAGRAM, SocialPlatform.TIKTOK]:
            return "CONTENT_FIRST_ENGAGEMENT"
        elif platform == SocialPlatform.LINKEDIN:
            return "PROFESSIONAL_RELATIONSHIP_BUILDING"
        elif platform == SocialPlatform.X:  # Updated from TWITTER
            return "REAL_TIME_CONVERSATION_ENGAGEMENT"
        else:
            return "VALUE_BASED_NURTURING"
