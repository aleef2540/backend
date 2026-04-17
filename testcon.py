import pymysql

conn = pymysql.connect(
    host="entstaffs.entraining.net",        # หรือ IP / domain
    user="entraini1_entrain",             # user db
    password="Ent.Pw78x.@a27df!z88",         # password db
    database="entraini1_entrainingdb", # ชื่อ database
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor
)

print("Connected!")