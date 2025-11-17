#!/usr/bin/env python3
"""
Script to fix deprecated datetime.utcnow() usage across the codebase.
Replaces datetime.utcnow() with datetime.now(datetime.UTC)
"""

import os
import re
from pathlib import Path

def fix_datetime_in_file(file_path: Path) -> tuple[bool, int]:
    """
    Fix datetime.utcnow() in a single file.
    Returns (was_modified, num_replacements)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Check if file already imports datetime
        has_datetime_import = bool(re.search(r'from datetime import.*datetime', content) or 
                                   re.search(r'import datetime', content))
        
        # Replace datetime.utcnow() with datetime.now(datetime.UTC)
        # This handles both `datetime.utcnow()` and `datetime.datetime.utcnow()`
        pattern = r'datetime\.utcnow\(\)'
        replacement = 'datetime.now(datetime.UTC)'
        
        content, num_replacements = re.subn(pattern, replacement, content)
        
        if num_replacements > 0:
            # Ensure datetime is imported if not already
            if not has_datetime_import:
                # Add import at the top after any existing imports
                lines = content.split('\n')
                import_index = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        import_index = i + 1
                    elif import_index > 0 and not line.startswith('import ') and not line.startswith('from ') and line.strip():
                        break
                
                if import_index == 0:
                    # No imports found, add at the top
                    lines.insert(0, 'from datetime import datetime')
                else:
                    lines.insert(import_index, 'from datetime import datetime')
                
                content = '\n'.join(lines)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, num_replacements
        
        return False, 0
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0

def main():
    """Main function to fix all Python files in backend/"""
    backend_dir = Path('backend')
    
    if not backend_dir.exists():
        print("Error: backend/ directory not found")
        return
    
    total_files_modified = 0
    total_replacements = 0
    
    # Find all Python files
    python_files = list(backend_dir.rglob('*.py'))
    
    print(f"Found {len(python_files)} Python files in backend/")
    print("Fixing datetime.utcnow() usage...\n")
    
    for file_path in python_files:
        was_modified, num_replacements = fix_datetime_in_file(file_path)
        
        if was_modified:
            total_files_modified += 1
            total_replacements += num_replacements
            print(f"✓ {file_path}: {num_replacements} replacement(s)")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Files modified: {total_files_modified}")
    print(f"  Total replacements: {total_replacements}")
    print(f"{'='*60}")
    
    if total_files_modified > 0:
        print("\n✅ All datetime.utcnow() calls have been replaced with datetime.now(datetime.UTC)")
    else:
        print("\n✅ No deprecated datetime.utcnow() calls found")

if __name__ == '__main__':
    main()

