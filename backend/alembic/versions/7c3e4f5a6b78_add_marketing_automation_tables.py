"""add marketing automation tables

Revision ID: 7c3e4f5a6b78
Revises: 6b2d3e4f5a67
Create Date: 2025-11-22 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7c3e4f5a6b78'
down_revision: Union[str, Sequence[str], None] = '6b2d3e4f5a67'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - add marketing automation tables."""
    
    # ========================================
    # 1. AUDIENCE SEGMENTS TABLE (Base table for campaigns)
    # ========================================
    op.create_table(
        'audience_segments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('criteria', sa.JSON(), nullable=False),
        sa.Column('operator', sa.String(length=10), nullable=False, server_default='and'),
        sa.Column('estimated_size', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_calculated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audience_segments_id', 'audience_segments', ['id'])
    op.create_index('ix_audience_segments_is_active', 'audience_segments', ['is_active'])
    
    # ========================================
    # 2. MARKETING TEMPLATES TABLE (Base table for campaigns)
    # ========================================
    op.create_table(
        'marketing_templates',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('template_type', sa.String(length=20), nullable=False),
        sa.Column('subject_line', sa.String(length=500), nullable=True),
        sa.Column('html_content', sa.Text(), nullable=True),
        sa.Column('text_content', sa.Text(), nullable=True),
        sa.Column('available_tokens', sa.JSON(), nullable=True),
        sa.Column('thumbnail_url', sa.String(length=500), nullable=True),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_marketing_templates_id', 'marketing_templates', ['id'])
    op.create_index('ix_marketing_templates_template_type', 'marketing_templates', ['template_type'])
    op.create_index('ix_marketing_templates_is_active', 'marketing_templates', ['is_active'])
    
    # ========================================
    # 3. AUTOMATION TRIGGERS TABLE (Base table for campaigns)
    # ========================================
    op.create_table(
        'automation_triggers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('trigger_type', sa.String(length=20), nullable=False),
        sa.Column('trigger_config', sa.JSON(), nullable=False),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_automation_triggers_id', 'automation_triggers', ['id'])
    op.create_index('ix_automation_triggers_trigger_type', 'automation_triggers', ['trigger_type'])
    op.create_index('ix_automation_triggers_is_active', 'automation_triggers', ['is_active'])
    
    # ========================================
    # 4. MARKETING CAMPAIGNS TABLE (References segments, templates, triggers)
    # ========================================
    op.create_table(
        'marketing_campaigns',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('campaign_type', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='draft'),
        sa.Column('segment_id', sa.Integer(), nullable=True),
        sa.Column('target_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('template_id', sa.Integer(), nullable=True),
        sa.Column('subject_line', sa.String(length=500), nullable=True),
        sa.Column('preview_text', sa.String(length=500), nullable=True),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_ab_test', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('ab_test_config', sa.JSON(), nullable=True),
        sa.Column('is_automated', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('automation_trigger_id', sa.Integer(), nullable=True),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['segment_id'], ['audience_segments.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['template_id'], ['marketing_templates.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['automation_trigger_id'], ['automation_triggers.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_marketing_campaigns_id', 'marketing_campaigns', ['id'])
    op.create_index('ix_marketing_campaigns_campaign_type', 'marketing_campaigns', ['campaign_type'])
    op.create_index('ix_marketing_campaigns_status', 'marketing_campaigns', ['status'])
    op.create_index('ix_marketing_campaigns_segment_id', 'marketing_campaigns', ['segment_id'])
    op.create_index('ix_marketing_campaigns_template_id', 'marketing_campaigns', ['template_id'])
    op.create_index('ix_marketing_campaigns_automation_trigger_id', 'marketing_campaigns', ['automation_trigger_id'])

    # ========================================
    # 5. CAMPAIGN ANALYTICS TABLE (References campaigns)
    # ========================================
    op.create_table(
        'campaign_analytics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('campaign_id', sa.Integer(), nullable=False),
        # Delivery Metrics
        sa.Column('total_sent', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_delivered', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_bounced', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_failed', sa.Integer(), nullable=False, server_default='0'),
        # Engagement Metrics
        sa.Column('total_opened', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('unique_opened', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_clicked', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('unique_clicked', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_unsubscribed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_spam_reports', sa.Integer(), nullable=False, server_default='0'),
        # Conversion Metrics
        sa.Column('total_conversions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_revenue', sa.Float(), nullable=False, server_default='0.0'),
        # Calculated Rates
        sa.Column('delivery_rate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('open_rate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('click_rate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('click_to_open_rate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('conversion_rate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('unsubscribe_rate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('roi', sa.Float(), nullable=False, server_default='0.0'),
        # Cost
        sa.Column('campaign_cost', sa.Float(), nullable=False, server_default='0.0'),
        # Timestamps
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['campaign_id'], ['marketing_campaigns.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('campaign_id')
    )
    op.create_index('ix_campaign_analytics_id', 'campaign_analytics', ['id'])
    op.create_index('ix_campaign_analytics_campaign_id', 'campaign_analytics', ['campaign_id'], unique=True)

    # ========================================
    # 6. CAMPAIGN SENDS TABLE (References campaigns)
    # ========================================
    op.create_table(
        'campaign_sends',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('campaign_id', sa.Integer(), nullable=False),
        # Recipient Info
        sa.Column('customer_id', sa.Integer(), nullable=True),
        sa.Column('recipient_email', sa.String(length=255), nullable=True),
        sa.Column('recipient_phone', sa.String(length=50), nullable=True),
        sa.Column('recipient_name', sa.String(length=255), nullable=True),
        # Send Status
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('sent_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('delivered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('opened_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('clicked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('converted_at', sa.DateTime(timezone=True), nullable=True),
        # Engagement Details
        sa.Column('open_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('click_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('links_clicked', sa.JSON(), nullable=True),
        # A/B Test Variant
        sa.Column('variant', sa.String(length=50), nullable=True),
        # Error Info
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('bounce_type', sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(['campaign_id'], ['marketing_campaigns.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_campaign_sends_id', 'campaign_sends', ['id'])
    op.create_index('ix_campaign_sends_campaign_id', 'campaign_sends', ['campaign_id'])
    op.create_index('ix_campaign_sends_customer_id', 'campaign_sends', ['customer_id'])
    op.create_index('ix_campaign_sends_status', 'campaign_sends', ['status'])
    op.create_index('ix_campaign_sends_recipient_email', 'campaign_sends', ['recipient_email'])


def downgrade() -> None:
    """Downgrade schema - remove marketing automation tables."""

    # Drop tables in reverse order (respecting foreign key constraints)

    # 6. Campaign Sends (references campaigns)
    op.drop_index('ix_campaign_sends_recipient_email', table_name='campaign_sends')
    op.drop_index('ix_campaign_sends_status', table_name='campaign_sends')
    op.drop_index('ix_campaign_sends_customer_id', table_name='campaign_sends')
    op.drop_index('ix_campaign_sends_campaign_id', table_name='campaign_sends')
    op.drop_index('ix_campaign_sends_id', table_name='campaign_sends')
    op.drop_table('campaign_sends')

    # 5. Campaign Analytics (references campaigns)
    op.drop_index('ix_campaign_analytics_campaign_id', table_name='campaign_analytics')
    op.drop_index('ix_campaign_analytics_id', table_name='campaign_analytics')
    op.drop_table('campaign_analytics')

    # 4. Marketing Campaigns (references segments, templates, triggers)
    op.drop_index('ix_marketing_campaigns_automation_trigger_id', table_name='marketing_campaigns')
    op.drop_index('ix_marketing_campaigns_template_id', table_name='marketing_campaigns')
    op.drop_index('ix_marketing_campaigns_segment_id', table_name='marketing_campaigns')
    op.drop_index('ix_marketing_campaigns_status', table_name='marketing_campaigns')
    op.drop_index('ix_marketing_campaigns_campaign_type', table_name='marketing_campaigns')
    op.drop_index('ix_marketing_campaigns_id', table_name='marketing_campaigns')
    op.drop_table('marketing_campaigns')

    # 3. Automation Triggers (base table)
    op.drop_index('ix_automation_triggers_is_active', table_name='automation_triggers')
    op.drop_index('ix_automation_triggers_trigger_type', table_name='automation_triggers')
    op.drop_index('ix_automation_triggers_id', table_name='automation_triggers')
    op.drop_table('automation_triggers')

    # 2. Marketing Templates (base table)
    op.drop_index('ix_marketing_templates_is_active', table_name='marketing_templates')
    op.drop_index('ix_marketing_templates_template_type', table_name='marketing_templates')
    op.drop_index('ix_marketing_templates_id', table_name='marketing_templates')
    op.drop_table('marketing_templates')

    # 1. Audience Segments (base table)
    op.drop_index('ix_audience_segments_is_active', table_name='audience_segments')
    op.drop_index('ix_audience_segments_id', table_name='audience_segments')
    op.drop_table('audience_segments')

