import sqlite3
import random

random.seed(42)

conn = sqlite3.connect("Bd5.db")

with conn, open("schema.sql") as schema:
    conn.executescript(schema.read())

with conn:
    for a0 in range(0, 1 << 11):
        print(a0)
        for a1 in range(0, 1 << 4):
            rand = random.randint(0xffff_ffff)
            conn.execute("INSERT INTO prefix (a0, a1, rand) VALUES (?, ?, ?)", (a0, a1, rand))
