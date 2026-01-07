#!/usr/bin/env python
"""Simple test script to check backend import"""
import sys
import os

# Disable bytecode
sys.dont_write_bytecode = True

print("Testing backend import...")
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

try:
    print("Importing backend.api.main...")
    import backend.api.main
    print("SUCCESS: Backend import OK")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

