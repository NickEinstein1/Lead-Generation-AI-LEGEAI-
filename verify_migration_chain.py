#!/usr/bin/env python3
"""Verify the Alembic migration chain is correct."""

import os
import re
from pathlib import Path

# Migration directory
migrations_dir = Path("backend/alembic/versions")

# Dictionary to store migration info
migrations = {}

# Read all migration files
for file_path in migrations_dir.glob("*.py"):
    if file_path.name == "__pycache__":
        continue
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract revision ID
    revision_match = re.search(r"revision\s*[:=]\s*['\"]([^'\"]+)['\"]", content)
    down_revision_match = re.search(r"down_revision\s*[:=]\s*['\"]([^'\"]+)['\"]", content)
    
    if revision_match:
        revision = revision_match.group(1)
        down_revision = down_revision_match.group(1) if down_revision_match else None
        
        # Extract description from docstring
        desc_match = re.search(r'"""([^\n]+)', content)
        description = desc_match.group(1) if desc_match else "Unknown"
        
        migrations[revision] = {
            'file': file_path.name,
            'description': description,
            'down_revision': down_revision,
            'children': []
        }

# Build the tree
for revision, info in migrations.items():
    if info['down_revision'] and info['down_revision'] in migrations:
        migrations[info['down_revision']]['children'].append(revision)

# Find the root (migration with no down_revision)
root = None
for revision, info in migrations.items():
    if info['down_revision'] is None or info['down_revision'] == 'None':
        root = revision
        break

# Print the migration chain
def print_chain(revision, indent=0):
    if revision not in migrations:
        return
    
    info = migrations[revision]
    prefix = "  " * indent
    arrow = "└─> " if indent > 0 else ""
    
    print(f"{prefix}{arrow}{revision[:12]} - {info['description']}")
    print(f"{prefix}    File: {info['file']}")
    
    if len(info['children']) > 1:
        print(f"{prefix}    ⚠️  WARNING: Multiple children detected (BRANCH)!")
    
    for child in info['children']:
        print_chain(child, indent + 1)

print("=" * 80)
print("ALEMBIC MIGRATION CHAIN")
print("=" * 80)
print()

if root:
    print_chain(root)
else:
    print("❌ ERROR: No root migration found!")

print()
print("=" * 80)
print("MIGRATION SUMMARY")
print("=" * 80)

# Check for issues
issues = []
for revision, info in migrations.items():
    if len(info['children']) > 1:
        issues.append(f"⚠️  Branch detected at {revision[:12]}: {len(info['children'])} children")

if issues:
    print("\n❌ ISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ No issues found - migration chain is linear!")

print()

