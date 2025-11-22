"""add crud tables

Revision ID: 6b2d3e4f5a67
Revises: file_doc_mgmt_001
Create Date: 2025-11-22 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6b2d3e4f5a67'
down_revision: Union[str, Sequence[str], None] = 'file_doc_mgmt_001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - add CRUD tables for customers, policies, claims, communications, and reports."""
    
    # ========================================
    # 1. CUSTOMERS TABLE
    # ========================================
    op.create_table(
        'customers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('phone', sa.String(length=50), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='active'),
        sa.Column('policies_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_value', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('join_date', sa.String(length=50), nullable=True),
        sa.Column('last_active', sa.String(length=50), nullable=True),
        sa.Column('reason', sa.String(length=255), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_customers_id', 'customers', ['id'])
    op.create_index('ix_customers_email', 'customers', ['email'])
    op.create_index('ix_customers_status', 'customers', ['status'])
    
    # ========================================
    # 2. POLICIES TABLE
    # ========================================
    op.create_table(
        'policies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('policy_number', sa.String(length=50), nullable=False),
        sa.Column('customer_id', sa.Integer(), nullable=True),
        sa.Column('customer_name', sa.String(length=255), nullable=False),
        sa.Column('policy_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='active'),
        sa.Column('premium', sa.String(length=50), nullable=False),
        sa.Column('coverage_amount', sa.Float(), nullable=True),
        sa.Column('start_date', sa.String(length=50), nullable=True),
        sa.Column('end_date', sa.String(length=50), nullable=True),
        sa.Column('renewal_date', sa.String(length=50), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('policy_number')
    )
    op.create_index('ix_policies_id', 'policies', ['id'])
    op.create_index('ix_policies_policy_number', 'policies', ['policy_number'], unique=True)
    op.create_index('ix_policies_customer_id', 'policies', ['customer_id'])
    op.create_index('ix_policies_policy_type', 'policies', ['policy_type'])
    op.create_index('ix_policies_status', 'policies', ['status'])
    
    # ========================================
    # 3. CLAIMS TABLE
    # ========================================
    op.create_table(
        'claims',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('claim_number', sa.String(length=50), nullable=False),
        sa.Column('policy_id', sa.Integer(), nullable=True),
        sa.Column('policy_number', sa.String(length=50), nullable=False),
        sa.Column('customer_name', sa.String(length=255), nullable=False),
        sa.Column('claim_type', sa.String(length=50), nullable=False),
        sa.Column('amount', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='pending'),
        sa.Column('claim_date', sa.String(length=50), nullable=True),
        sa.Column('due_date', sa.String(length=50), nullable=True),
        sa.Column('processed_date', sa.String(length=50), nullable=True),
        sa.Column('description', sa.String(length=1000), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['policy_id'], ['policies.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('claim_number')
    )
    op.create_index('ix_claims_id', 'claims', ['id'])
    op.create_index('ix_claims_claim_number', 'claims', ['claim_number'], unique=True)
    op.create_index('ix_claims_policy_id', 'claims', ['policy_id'])
    op.create_index('ix_claims_claim_type', 'claims', ['claim_type'])
    op.create_index('ix_claims_status', 'claims', ['status'])
    
    # ========================================
    # 4. COMMUNICATIONS TABLE
    # ========================================
    op.create_table(
        'communications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('customer_id', sa.Integer(), nullable=True),
        sa.Column('customer_name', sa.String(length=255), nullable=False),
        sa.Column('comm_type', sa.String(length=50), nullable=False),
        sa.Column('channel', sa.String(length=50), nullable=False),
        sa.Column('subject', sa.String(length=500), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='sent'),
        sa.Column('comm_date', sa.String(length=50), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_communications_id', 'communications', ['id'])
    op.create_index('ix_communications_customer_id', 'communications', ['customer_id'])
    op.create_index('ix_communications_comm_type', 'communications', ['comm_type'])
    op.create_index('ix_communications_status', 'communications', ['status'])

    # ========================================
    # 5. REPORTS TABLE
    # ========================================
    op.create_table(
        'reports',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('report_number', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('report_type', sa.String(length=100), nullable=False),
        sa.Column('period', sa.String(length=100), nullable=False),
        sa.Column('format', sa.String(length=50), nullable=False, server_default='PDF'),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='completed'),
        sa.Column('generated_date', sa.String(length=50), nullable=True),
        sa.Column('data', sa.JSON(), nullable=True),
        sa.Column('file_path', sa.String(length=500), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('report_number')
    )
    op.create_index('ix_reports_id', 'reports', ['id'])
    op.create_index('ix_reports_report_number', 'reports', ['report_number'], unique=True)
    op.create_index('ix_reports_report_type', 'reports', ['report_type'])
    op.create_index('ix_reports_status', 'reports', ['status'])


def downgrade() -> None:
    """Downgrade schema - remove CRUD tables."""

    # Drop tables in reverse order (respecting foreign key constraints)

    # 5. Reports (no foreign keys)
    op.drop_index('ix_reports_status', table_name='reports')
    op.drop_index('ix_reports_report_type', table_name='reports')
    op.drop_index('ix_reports_report_number', table_name='reports')
    op.drop_index('ix_reports_id', table_name='reports')
    op.drop_table('reports')

    # 4. Communications (references customers)
    op.drop_index('ix_communications_status', table_name='communications')
    op.drop_index('ix_communications_comm_type', table_name='communications')
    op.drop_index('ix_communications_customer_id', table_name='communications')
    op.drop_index('ix_communications_id', table_name='communications')
    op.drop_table('communications')

    # 3. Claims (references policies)
    op.drop_index('ix_claims_status', table_name='claims')
    op.drop_index('ix_claims_claim_type', table_name='claims')
    op.drop_index('ix_claims_policy_id', table_name='claims')
    op.drop_index('ix_claims_claim_number', table_name='claims')
    op.drop_index('ix_claims_id', table_name='claims')
    op.drop_table('claims')

    # 2. Policies (references customers)
    op.drop_index('ix_policies_status', table_name='policies')
    op.drop_index('ix_policies_policy_type', table_name='policies')
    op.drop_index('ix_policies_customer_id', table_name='policies')
    op.drop_index('ix_policies_policy_number', table_name='policies')
    op.drop_index('ix_policies_id', table_name='policies')
    op.drop_table('policies')

    # 1. Customers (base table)
    op.drop_index('ix_customers_status', table_name='customers')
    op.drop_index('ix_customers_email', table_name='customers')
    op.drop_index('ix_customers_id', table_name='customers')
    op.drop_table('customers')

