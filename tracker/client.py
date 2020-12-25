import sys
import requests

VERSION = 1
ENDPOINT = sys.argv[1].rstrip("/")
SECRET = sys.argv[2]

while True:
    res = requests.post(ENDPOINT + "/acquire", json={
        "key": SECRET,
        "version": VERSION,
    })
    res.raise_for_status()
    res = res.json()
    print(res)
