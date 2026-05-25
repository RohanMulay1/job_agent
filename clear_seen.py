import sqlite3
conn = sqlite3.connect("data/jobs.db")
for status in ("seen", "skipped_external", "skipped_fit"):
    cur = conn.execute(f"DELETE FROM jobs WHERE status = '{status}'")
    print(f"Deleted {cur.rowcount} {status} records")
conn.commit()
stats = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
print("Remaining:", stats)
conn.close()
