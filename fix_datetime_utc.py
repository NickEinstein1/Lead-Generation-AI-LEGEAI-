#!/usr/bin/env python3
"""
Fix datetime.UTC to timezone.utc for Python 3.10+ compatibility.
datetime.UTC was added in Python 3.11, but timezone.utc works in all versions.
"""

import os
import re
from pathlib import Path

def fix_datetime_utc(file_path):
    """Replace datetime.UTC with timezone.utc and ensure timezone is imported"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Check if file uses datetime.UTC
    if 'datetime.UTC' not in content:
        return False
    
    # Replace datetime.UTC with timezone.utc
    content = re.sub(
        r'datetime\.UTC',
        'timezone.utc',
        content
    )
    
    # Ensure timezone is imported
    # Check if datetime is imported
    if 'from datetime import' in content:
        # Check if timezone is already imported
        if not re.search(r'from datetime import.*timezone', content):
            # Add timezone to the import
            content = re.sub(
                r'(from datetime import [^;\n]+)',
                lambda m: m.group(1) + ', timezone' if 'timezone' not in m.group(1) else m.group(1),
                content
            )
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    backend_dir = Path('backend')
    
    if not backend_dir.exists():
        print("Error: backend directory not found")
        return
    
    fixed_files = []
    total_replacements = 0
    
    # Find all Python files
    for py_file in backend_dir.rglob('*.py'):
        if fix_datetime_utc(py_file):
            fixed_files.append(py_file)
            # Count replacements
            with open(py_file, 'r') as f:
                count = f.read().count('timezone.utc')
            total_replacements += count
    
    print(f"✅ Fixed {len(fixed_files)} files with {total_replacements} timezone.utc usages")
    print("\nFixed files:")
    for f in sorted(fixed_files):
        print(f"  - {f}")

if __name__ == '__main__':
    main()

