"""add scores table

Revision ID: 137b19ac6ef3
Revises: 2dd19a2d0626
Create Date: 2025-10-21 06:45:39.526399

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '137b19ac6ef3'
down_revision: Union[str, Sequence[str], None] = '2dd19a2d0626'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'scores',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('lead_id', sa.Integer(), sa.ForeignKey('leads.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('score', sa.Float()),
        sa.Column('band', sa.String(length=32)),
        sa.Column('explanation', sa.JSON()),
        sa.Column('model_version', sa.String(length=64)),
        sa.Column('features', sa.JSON()),
        sa.Column('scored_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    )
    op.create_index('ix_scores_lead_id', 'scores', ['lead_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_scores_lead_id', table_name='scores')
    op.drop_table('scores')
