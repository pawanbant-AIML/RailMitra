import sqlite3

DB = r"C:\fare\train-ticket-assistant\scripts\train_ticket.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", [t[0] for t in tables])
print()

for t in tables:
    name = t[0]
    count = cur.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
    print(f"  {name}: {count} rows")

print()
print("Sample trains:")
for r in cur.execute("SELECT * FROM trains LIMIT 5").fetchall():
    print(" ", r)

print()
print("Sample stations:")
for r in cur.execute("SELECT * FROM stations LIMIT 5").fetchall():
    print(" ", r)

print()
print("Sample fares:")
try:
    for r in cur.execute("SELECT * FROM fares LIMIT 5").fetchall():
        print(" ", r)
except Exception as e:
    print(f"  (no fares table: {e})")

print()
print("Sample routes:")
try:
    for r in cur.execute("SELECT * FROM routes LIMIT 5").fetchall():
        print(" ", r)
except Exception as e:
    print(f"  (no routes table: {e})")

print()
print("Bangalore trains sample:")
for r in cur.execute("SELECT * FROM trains WHERE source_station_code IN ('SBC','BNC','BAND','YPR') OR destination_station_code IN ('SBC','BNC','BAND','YPR') LIMIT 5").fetchall():
    print(" ", r)

print()
print("Station search 'bangalore':")
for r in cur.execute("SELECT * FROM stations WHERE LOWER(station_name) LIKE '%bangal%' OR LOWER(station_code) LIKE '%sbc%'").fetchall():
    print(" ", r)

print()
print("Station search 'mumbai':")
for r in cur.execute("SELECT * FROM stations WHERE LOWER(station_name) LIKE '%mumbai%' OR LOWER(station_code) LIKE '%csmt%' OR LOWER(station_code) LIKE '%bct%'").fetchall():
    print(" ", r)

conn.close()
