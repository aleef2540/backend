import sqlite3
import csv
from pathlib import Path

DB_FILE = "ai_idp_script.db"
CSV_FILE = "ai_idp_script.csv"

# =========================
# 1) เช็คไฟล์ CSV
# =========================
if not Path(CSV_FILE).exists():
    print(f"[ERROR] ไม่พบไฟล์ {CSV_FILE}")
    exit()

# =========================
# 2) สร้าง DB + Table
# =========================
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS ai_idp_script (
    script_id TEXT PRIMARY KEY,
    vdo_name TEXT,
    course_name TEXT,
    youtubelink TEXT,
    script TEXT
)
""")

conn.commit()

# =========================
# 3) อ่าน CSV + Insert
# =========================
insert_count = 0
skip_count = 0

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    # debug header
    print("[CSV HEADER] =", reader.fieldnames)

    for row in reader:
        try:
            cur.execute("""
            INSERT OR IGNORE INTO ai_idp_script (
                script_id,
                vdo_name,
                course_name,
                youtubelink,
                script
            ) VALUES (?, ?, ?, ?, ?)
            """, (
                row.get("script_id", "").strip(),
                row.get("vdo_name", "").strip(),
                row.get("course_name", "").strip(),
                row.get("youtubelink", "").strip(),
                row.get("script", "").strip(),
            ))
            insert_count += 1
        except Exception as e:
            skip_count += 1
            print("[SKIP ROW ERROR]", e)

conn.commit()

# =========================
# 4) เช็คผลลัพธ์
# =========================
cur.execute("SELECT COUNT(*) FROM ai_idp_script")
total = cur.fetchone()[0]

conn.close()

print("=================================")
print("Import เสร็จแล้ว")
print(f"Insert Attempt: {insert_count}")
print(f"Skip: {skip_count}")
print(f"Total Rows in DB: {total}")
print("DB FILE:", DB_FILE)
print("=================================")