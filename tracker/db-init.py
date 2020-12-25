import sqlite3

conn = sqlite3.connect("Bd5.db")

with conn, open("schema.sql") as schema:
    conn.executescript(schema.read())

with conn:
    for a0 in range(0, 1 << 11):
        print(a)
        for a1 in range(0, 1 << 4):
            conn.execute("INSERT INTO prefix (a0, a1) VALUES (?, ?)", (a0, a1))
