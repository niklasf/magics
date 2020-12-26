import sqlite3
import random

random.seed(42)

conn = sqlite3.connect("Rh1.db")

with conn, open("schema.sql") as schema:
    conn.executescript(schema.read())

# 0b_0_0_0_0_0_000000000_00000000_00000000_00000100_00000010_00000001_000000001

with conn:
    for a0 in range(0, 1 << 9):
        print(a0)
        for a1 in range(0, 1 << 8):
            for a2 in range(0, 1 << 8):
                rand = random.randint(0xffff_ffff)
                conn.execute("INSERT INTO prefix (a0, a1, a2, rand) VALUES (?, ?, ?, ?)", (a0, a1, a2, rand))

