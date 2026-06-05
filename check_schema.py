import sqlite3
DB = r"C:\fare\train-ticket-assistant\scripts\train_ticket.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

for t in ['trains', 'stations', 'routes', 'fares', 'bookings']:
    try:
        info = cur.execute(f"PRAGMA table_info([{t}])").fetchall()
        print(f"\n=== {t} ===")
        for col in info:
            print(f"  {col}")
    except Exception as e:
        print(f"\n=== {t} === ERROR: {e}")
conn.close()
