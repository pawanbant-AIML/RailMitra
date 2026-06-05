import sqlite3

DB = r"C:\fare\train-ticket-assistant\scripts\train_ticket.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

# Find trains from SBC -> CSTM or BCT or LTT or DR
print("=== Direct trains SBC -> Mumbai stations ===")
for r in cur.execute("""
    SELECT train_number, train_name, source_station_code, destination_station_code
    FROM trains
    WHERE source_station_code = 'SBC'
      AND destination_station_code IN ('CSTM','BCT','LTT','DR','BDTS')
""").fetchall():
    print(" ", r)

print()
print("=== Direct trains YPR -> Mumbai stations ===")
for r in cur.execute("""
    SELECT train_number, train_name, source_station_code, destination_station_code
    FROM trains
    WHERE source_station_code = 'YPR'
      AND destination_station_code IN ('CSTM','BCT','LTT','DR','BDTS')
""").fetchall():
    print(" ", r)

print()
print("=== Route-based SBC/YPR to CSTM/BCT/LTT ===")
# Trains with both SBC and CSTM in their route
for r in cur.execute("""
    SELECT DISTINCT r1.train_number
    FROM routes r1
    JOIN routes r2 ON r1.train_number = r2.train_number
    WHERE r1.station_code IN ('SBC','YPR')
      AND r2.station_code IN ('CSTM','BCT','LTT','DR','BDTS')
      AND r1.sequence < r2.sequence
    LIMIT 10
""").fetchall():
    tn = r[0]
    train = cur.execute("SELECT train_number, train_name, source_station_code, destination_station_code FROM trains WHERE train_number=?", (tn,)).fetchone()
    if train:
        print(" ", train)

conn.close()
