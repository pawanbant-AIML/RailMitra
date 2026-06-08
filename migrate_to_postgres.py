#!/usr/bin/env python
"""
Migrate SQLite database to PostgreSQL on Render.
Reads from scripts/train_ticket.db and writes to Render PostgreSQL.
"""

import os
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / "backend" / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Get database URLs
SQLITE_DB = str(Path(__file__).parent / "scripts" / "train_ticket.db")
POSTGRES_URL = os.getenv("DATABASE_URL")

if not POSTGRES_URL:
    print("❌ ERROR: DATABASE_URL not set in .env")
    sys.exit(1)

if not Path(SQLITE_DB).exists():
    print(f"❌ ERROR: {SQLITE_DB} not found")
    sys.exit(1)

print(f"📊 SQLite Source: {SQLITE_DB}")
print(f"🗄️  PostgreSQL Target: {POSTGRES_URL.split('@')[1]}")
print()

# Create engines
sqlite_engine = create_engine(f"sqlite:///{SQLITE_DB}")
postgres_engine = create_engine(POSTGRES_URL)

# Get all tables from SQLite
inspector = inspect(sqlite_engine)
tables = inspector.get_table_names()

print(f"📋 Tables to migrate: {tables}")
print()

try:
    # Create tables in PostgreSQL first
    print("🔧 Creating table schemas in PostgreSQL...")
    from app.models.train_models import Base
    Base.metadata.create_all(postgres_engine)
    print("✅ Schemas created\n")

    # Migrate data table by table
    for table in tables:
        print(f"📤 Migrating table: {table}")
        
        try:
            # Read from SQLite
            df = pd.read_sql_table(table, sqlite_engine)
            
            if df.empty:
                print(f"   ⚠️  Table is empty, skipping...")
                continue
            
            # Write to PostgreSQL
            df.to_sql(table, postgres_engine, if_exists="append", index=False)
            print(f"   ✅ Migrated {len(df)} rows to {table}")
            
        except Exception as e:
            print(f"   ❌ Error migrating {table}: {e}")

    print("\n✅ Migration complete!")
    print("\n📊 Summary:")
    
    # Verify migration
    with postgres_engine.connect() as conn:
        for table in tables:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = result.scalar()
            print(f"   • {table}: {count} rows")

except Exception as e:
    print(f"❌ Migration failed: {e}")
    sys.exit(1)
