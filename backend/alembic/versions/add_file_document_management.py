"""add file document management

Revision ID: file_doc_mgmt_001
Revises: 5a1c2e3d4f56
Create Date: 2025-11-17 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'file_doc_mgmt_001'
down_revision = '5a1c2e3d4f56'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade schema - add file document management tables"""
    
    # Create document_categories table
    op.create_table(
        'document_categories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('display_name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('icon', sa.String(length=50), nullable=True),
        sa.Column('color', sa.String(length=20), nullable=True),
        sa.Column('parent_category_id', sa.Integer(), nullable=True),
        sa.Column('sort_order', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['parent_category_id'], ['document_categories.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('ix_document_categories_id', 'document_categories', ['id'])
    
    # Create file_documents table
    op.create_table(
        'file_documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('file_type', sa.String(length=50), nullable=False),
        sa.Column('mime_type', sa.String(length=100), nullable=False),
        sa.Column('file_size', sa.BigInteger(), nullable=False),
        sa.Column('file_path', sa.String(length=512), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('lead_id', sa.Integer(), nullable=True),
        sa.Column('uploaded_by', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='active'),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('parent_document_id', sa.Integer(), nullable=True),
        sa.Column('is_public', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('access_level', sa.String(length=32), nullable=False, server_default='private'),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('last_accessed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['lead_id'], ['leads.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['uploaded_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['parent_document_id'], ['file_documents.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_file_documents_id', 'file_documents', ['id'])
    op.create_index('ix_file_documents_category', 'file_documents', ['category'])
    op.create_index('ix_file_documents_lead_id', 'file_documents', ['lead_id'])
    op.create_index('ix_file_documents_uploaded_by', 'file_documents', ['uploaded_by'])
    op.create_index('ix_file_documents_status', 'file_documents', ['status'])
    op.create_index('ix_file_documents_category_status', 'file_documents', ['category', 'status'])
    op.create_index('ix_file_documents_file_type', 'file_documents', ['file_type'])
    
    # Create document_shares table
    op.create_table(
        'document_shares',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('shared_with_user_id', sa.Integer(), nullable=True),
        sa.Column('shared_by_user_id', sa.Integer(), nullable=True),
        sa.Column('can_view', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('can_download', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('can_edit', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('can_delete', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('can_share', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('share_token', sa.String(length=255), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('access_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_accessed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['file_documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['shared_with_user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['shared_by_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('share_token')
    )
    op.create_index('ix_document_shares_id', 'document_shares', ['id'])
    op.create_index('ix_document_shares_document_id', 'document_shares', ['document_id'])
    op.create_index('ix_document_shares_shared_with_user_id', 'document_shares', ['shared_with_user_id'])
    op.create_index('ix_document_shares_token', 'document_shares', ['share_token'])
    op.create_index('ix_document_shares_document_user', 'document_shares', ['document_id', 'shared_with_user_id'])
    
    # Create document_activities table
    op.create_table(
        'document_activities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('activity_type', sa.String(length=50), nullable=False),
        sa.Column('activity_details', sa.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=512), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['file_documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_document_activities_id', 'document_activities', ['id'])
    op.create_index('ix_document_activities_document_id', 'document_activities', ['document_id'])
    op.create_index('ix_document_activities_user_id', 'document_activities', ['user_id'])
    op.create_index('ix_document_activities_activity_type', 'document_activities', ['activity_type'])
    op.create_index('ix_document_activities_type_created', 'document_activities', ['activity_type', 'created_at'])
    op.create_index('ix_document_activities_document_created', 'document_activities', ['document_id', 'created_at'])

    # Insert default categories
    op.execute("""
        INSERT INTO document_categories (name, display_name, description, icon, color, sort_order, is_active)
        VALUES
            ('policies', 'Insurance Policies', 'Insurance policy documents', '📋', '#3B82F6', 1, true),
            ('claims', 'Claims', 'Insurance claim documents', '📝', '#EF4444', 2, true),
            ('customer_data', 'Customer Data', 'Customer information and data files', '👥', '#10B981', 3, true),
            ('contracts', 'Contracts', 'Legal contracts and agreements', '📄', '#8B5CF6', 4, true),
            ('reports', 'Reports', 'Business reports and analytics', '📊', '#F59E0B', 5, true),
            ('correspondence', 'Correspondence', 'Email and letter correspondence', '✉️', '#6366F1', 6, true),
            ('financial', 'Financial Documents', 'Financial statements and records', '💰', '#059669', 7, true),
            ('other', 'Other', 'Miscellaneous documents', '📁', '#6B7280', 99, true)
    """)


def downgrade() -> None:
    """Downgrade schema - remove file document management tables"""
    op.drop_index('ix_document_activities_document_created', table_name='document_activities')
    op.drop_index('ix_document_activities_type_created', table_name='document_activities')
    op.drop_index('ix_document_activities_activity_type', table_name='document_activities')
    op.drop_index('ix_document_activities_user_id', table_name='document_activities')
    op.drop_index('ix_document_activities_document_id', table_name='document_activities')
    op.drop_index('ix_document_activities_id', table_name='document_activities')
    op.drop_table('document_activities')

    op.drop_index('ix_document_shares_document_user', table_name='document_shares')
    op.drop_index('ix_document_shares_token', table_name='document_shares')
    op.drop_index('ix_document_shares_shared_with_user_id', table_name='document_shares')
    op.drop_index('ix_document_shares_document_id', table_name='document_shares')
    op.drop_index('ix_document_shares_id', table_name='document_shares')
    op.drop_table('document_shares')

    op.drop_index('ix_file_documents_file_type', table_name='file_documents')
    op.drop_index('ix_file_documents_category_status', table_name='file_documents')
    op.drop_index('ix_file_documents_status', table_name='file_documents')
    op.drop_index('ix_file_documents_uploaded_by', table_name='file_documents')
    op.drop_index('ix_file_documents_lead_id', table_name='file_documents')
    op.drop_index('ix_file_documents_category', table_name='file_documents')
    op.drop_index('ix_file_documents_id', table_name='file_documents')
    op.drop_table('file_documents')

    op.drop_index('ix_document_categories_id', table_name='document_categories')
    op.drop_table('document_categories')

