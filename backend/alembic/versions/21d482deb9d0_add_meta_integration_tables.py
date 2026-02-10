"""Add Meta integration tables

Revision ID: 21d482deb9d0
Revises: 7c3e4f5a6b78
Create Date: 2026-02-05 14:42:12.732259

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '21d482deb9d0'
down_revision: Union[str, Sequence[str], None] = '7c3e4f5a6b78'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create meta_integrations table
    op.create_table(
        'meta_integrations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('access_token', sa.Text(), nullable=False),
        sa.Column('token_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('meta_user_id', sa.String(length=100), nullable=True),
        sa.Column('meta_user_name', sa.String(length=255), nullable=True),
        sa.Column('scopes', sa.ARRAY(sa.String()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_meta_integrations_id'), 'meta_integrations', ['id'], unique=False)
    op.create_index(op.f('ix_meta_integrations_user_id'), 'meta_integrations', ['user_id'], unique=False)

    # Create meta_ad_accounts table
    op.create_table(
        'meta_ad_accounts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('integration_id', sa.Integer(), nullable=False),
        sa.Column('ad_account_id', sa.String(length=100), nullable=False),
        sa.Column('ad_account_name', sa.String(length=255), nullable=True),
        sa.Column('currency', sa.String(length=10), nullable=True),
        sa.Column('timezone', sa.String(length=100), nullable=True),
        sa.Column('account_status', sa.String(length=50), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['integration_id'], ['meta_integrations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('ad_account_id')
    )
    op.create_index(op.f('ix_meta_ad_accounts_id'), 'meta_ad_accounts', ['id'], unique=False)
    op.create_index(op.f('ix_meta_ad_accounts_ad_account_id'), 'meta_ad_accounts', ['ad_account_id'], unique=True)

    # Create meta_pages table
    op.create_table(
        'meta_pages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('integration_id', sa.Integer(), nullable=False),
        sa.Column('page_id', sa.String(length=100), nullable=False),
        sa.Column('page_name', sa.String(length=255), nullable=True),
        sa.Column('page_access_token', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('followers_count', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['integration_id'], ['meta_integrations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('page_id')
    )
    op.create_index(op.f('ix_meta_pages_id'), 'meta_pages', ['id'], unique=False)
    op.create_index(op.f('ix_meta_pages_page_id'), 'meta_pages', ['page_id'], unique=True)

    # Create meta_lead_forms table
    op.create_table(
        'meta_lead_forms',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('page_id', sa.Integer(), nullable=False),
        sa.Column('form_id', sa.String(length=100), nullable=False),
        sa.Column('form_name', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('leads_count', sa.Integer(), nullable=True, default=0),
        sa.Column('auto_sync', sa.Boolean(), nullable=True, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['page_id'], ['meta_pages.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('form_id')
    )
    op.create_index(op.f('ix_meta_lead_forms_id'), 'meta_lead_forms', ['id'], unique=False)
    op.create_index(op.f('ix_meta_lead_forms_form_id'), 'meta_lead_forms', ['form_id'], unique=True)

    # Create meta_lead_mappings table
    op.create_table(
        'meta_lead_mappings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('meta_lead_id', sa.String(length=100), nullable=False),
        sa.Column('legeai_lead_id', sa.Integer(), nullable=False),
        sa.Column('form_id', sa.String(length=100), nullable=True),
        sa.Column('page_id', sa.String(length=100), nullable=True),
        sa.Column('synced_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('meta_lead_id')
    )
    op.create_index(op.f('ix_meta_lead_mappings_id'), 'meta_lead_mappings', ['id'], unique=False)
    op.create_index(op.f('ix_meta_lead_mappings_meta_lead_id'), 'meta_lead_mappings', ['meta_lead_id'], unique=True)
    op.create_index(op.f('ix_meta_lead_mappings_legeai_lead_id'), 'meta_lead_mappings', ['legeai_lead_id'], unique=False)

    # Create meta_campaigns table
    op.create_table(
        'meta_campaigns',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('campaign_id', sa.String(length=100), nullable=False),
        sa.Column('ad_account_id', sa.String(length=100), nullable=False),
        sa.Column('campaign_name', sa.String(length=255), nullable=False),
        sa.Column('objective', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('daily_budget', sa.Integer(), nullable=True),
        sa.Column('lifetime_budget', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('campaign_id')
    )
    op.create_index(op.f('ix_meta_campaigns_id'), 'meta_campaigns', ['id'], unique=False)
    op.create_index(op.f('ix_meta_campaigns_campaign_id'), 'meta_campaigns', ['campaign_id'], unique=True)

    # Create meta_catalogs table
    op.create_table(
        'meta_catalogs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('catalog_id', sa.String(length=100), nullable=False),
        sa.Column('business_id', sa.String(length=100), nullable=False),
        sa.Column('catalog_name', sa.String(length=255), nullable=False),
        sa.Column('vertical', sa.String(length=50), nullable=True),
        sa.Column('product_count', sa.Integer(), nullable=True, default=0),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('catalog_id')
    )
    op.create_index(op.f('ix_meta_catalogs_id'), 'meta_catalogs', ['id'], unique=False)
    op.create_index(op.f('ix_meta_catalogs_catalog_id'), 'meta_catalogs', ['catalog_id'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables in reverse order
    op.drop_index(op.f('ix_meta_catalogs_catalog_id'), table_name='meta_catalogs')
    op.drop_index(op.f('ix_meta_catalogs_id'), table_name='meta_catalogs')
    op.drop_table('meta_catalogs')

    op.drop_index(op.f('ix_meta_campaigns_campaign_id'), table_name='meta_campaigns')
    op.drop_index(op.f('ix_meta_campaigns_id'), table_name='meta_campaigns')
    op.drop_table('meta_campaigns')

    op.drop_index(op.f('ix_meta_lead_mappings_legeai_lead_id'), table_name='meta_lead_mappings')
    op.drop_index(op.f('ix_meta_lead_mappings_meta_lead_id'), table_name='meta_lead_mappings')
    op.drop_index(op.f('ix_meta_lead_mappings_id'), table_name='meta_lead_mappings')
    op.drop_table('meta_lead_mappings')

    op.drop_index(op.f('ix_meta_lead_forms_form_id'), table_name='meta_lead_forms')
    op.drop_index(op.f('ix_meta_lead_forms_id'), table_name='meta_lead_forms')
    op.drop_table('meta_lead_forms')

    op.drop_index(op.f('ix_meta_pages_page_id'), table_name='meta_pages')
    op.drop_index(op.f('ix_meta_pages_id'), table_name='meta_pages')
    op.drop_table('meta_pages')

    op.drop_index(op.f('ix_meta_ad_accounts_ad_account_id'), table_name='meta_ad_accounts')
    op.drop_index(op.f('ix_meta_ad_accounts_id'), table_name='meta_ad_accounts')
    op.drop_table('meta_ad_accounts')

    op.drop_index(op.f('ix_meta_integrations_user_id'), table_name='meta_integrations')
    op.drop_index(op.f('ix_meta_integrations_id'), table_name='meta_integrations')
    op.drop_table('meta_integrations')
