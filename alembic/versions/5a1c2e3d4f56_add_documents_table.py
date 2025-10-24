"""add documents table

Revision ID: 5a1c2e3d4f56
Revises: 137b19ac6ef3
Create Date: 2025-10-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '5a1c2e3d4f56'
down_revision: Union[str, Sequence[str], None] = '137b19ac6ef3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('lead_id', sa.Integer(), sa.ForeignKey('leads.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=32), server_default='pending', nullable=False),
        sa.Column('provider', sa.String(length=64), server_default='internal', nullable=True),
        sa.Column('provider_request_id', sa.String(length=255), nullable=True),
        sa.Column('signing_url', sa.String(length=512), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('signed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    )
    op.create_index('ix_documents_lead_id', 'documents', ['lead_id'])


def downgrade() -> None:
    op.drop_index('ix_documents_lead_id', table_name='documents')
    op.drop_table('documents')

