#!/usr/bin/env python3
"""
Migrate SQLite database to PostgreSQL on Render.

Reads from:
    scripts/train_ticket.db

Writes to:
    PostgreSQL DATABASE_URL from backend/.env
"""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text


def normalize_postgres_url(url: str) -> str:
    """
    Make Render/PostgreSQL URLs SQLAlchemy-friendly.

    - postgres:// -> postgresql+psycopg2://
    - Ensure sslmode=require is present
    """
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)

    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))

    if "sslmode" not in query:
        query["sslmode"] = "require"

    parsed = parsed._replace(query=urlencode(query))
    return urlunparse(parsed)


def get_table_row_count(conn, table_name: str) -> int:
    """Return row count for a table in PostgreSQL."""
    result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
    return int(result.scalar() or 0)


def main() -> int:
    # Setup paths
    base_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(base_dir / "backend"))

    # Load environment variables
    env_path = base_dir / "backend" / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Get database URLs
    sqlite_db = base_dir / "scripts" / "train_ticket.db"
    postgres_url = os.getenv("DATABASE_URL")

    if not postgres_url:
        print("❌ ERROR: DATABASE_URL not set in backend/.env")
        return 1

    if not sqlite_db.exists():
        print(f"❌ ERROR: {sqlite_db} not found")
        return 1

    postgres_url = normalize_postgres_url(postgres_url)

    print(f"📊 SQLite Source: {sqlite_db}")
    print(f"🗄️  PostgreSQL Target: {postgres_url.split('@')[1] if '@' in postgres_url else postgres_url}")
    print()

    # Create engines
    sqlite_engine = create_engine(f"sqlite:///{sqlite_db}")
    postgres_engine = create_engine(postgres_url)

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
            print(f"📤 Processing table: {table}")

            try:
                # Read from SQLite
                df = pd.read_sql_table(table, sqlite_engine)

                if df.empty:
                    print("   ⚠️  Table is empty, skipping...")
                    continue

                # Check if PostgreSQL already has data for this table
                with postgres_engine.connect() as conn:
                    existing_count = get_table_row_count(conn, table)

                if existing_count > 0:
                    print(f"   ✅ PostgreSQL already has {existing_count:,} rows in {table}, skipping insert...")
                    continue

                # Write to PostgreSQL
                df.to_sql(
                    table,
                    postgres_engine,
                    if_exists="append",
                    index=False,
                    chunksize=1000,
                    method="multi",
                )
                print(f"   ✅ Migrated {len(df):,} rows to {table}")

            except Exception as e:
                print(f"   ❌ Error processing {table}: {e}")

        print("\n✅ Migration step complete!")
        print("\n📊 PostgreSQL Verification:")

        # Verify migration
        with postgres_engine.connect() as conn:
            for table in tables:
                try:
                    count = get_table_row_count(conn, table)
                    print(f"   • {table}: {count:,} rows")
                except Exception as e:
                    print(f"   • {table}: verification failed ({e})")

        return 0

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())