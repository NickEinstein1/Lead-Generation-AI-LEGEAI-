#!/usr/bin/env python
"""Test database connection to Supabase"""
import sys
sys.dont_write_bytecode = True

import asyncio
from dotenv import load_dotenv
load_dotenv()

import os
print("Testing Supabase Connection...")
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')[:80]}...")

async def test_connection():
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        
        DATABASE_URL = os.getenv("DATABASE_URL")
        engine = create_async_engine(DATABASE_URL, echo=False)
        
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            print(f"Database query result: {row}")
            print("SUCCESS: Database connection works!")

            # List tables
            result = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [r[0] for r in result.fetchall()]
            print(f"Tables in database: {tables}")

        await engine.dispose()
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())

