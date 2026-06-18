import os
from sqlalchemy import create_engine, text

# Use the EXTERNAL Database URL from Render
DATABASE_URL = 'postgresql://railmitra_jvr9_user:vWHmX1BPDusmUwVUUjvKzkIanPxeD7uu@dpg-d8heu342m8qs73b2v1cg-a.oregon-postgres.render.com/railmitra_jvr9'

print("Connecting to database...")
engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        # Check if trains table exists
        result = conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'trains')"))
        table_exists = result.scalar()
        print(f"Trains table exists: {table_exists}")

        if table_exists:
            # Count trains
            result = conn.execute(text("SELECT COUNT(*) FROM trains"))
            count = result.scalar()
            print(f"Number of trains: {count}")

            # Sample 5 trains
            result = conn.execute(text("SELECT train_number, train_name FROM trains LIMIT 5"))
            print("Sample trains:")
            for row in result:
                print(f"  {row[0]}: {row[1]}")

            # Check routes table
            result = conn.execute(text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'routes')"))
            routes_table = result.scalar()
            print(f"Routes table exists: {routes_table}")
            if routes_table:
                result = conn.execute(text("SELECT COUNT(*) FROM routes"))
                print(f"Number of route entries: {result.scalar()}")
        else:
            print("No trains table found! Did you run migrations?")
            print("Check if the database schema is created.")
except Exception as e:
    print(f"Error connecting to database: {e}")