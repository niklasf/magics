import sys
import requests
import subprocess

VERSION = 4
ENDPOINT = sys.argv[1].rstrip("/")
SECRET = sys.argv[2]

while True:
    res = requests.post(ENDPOINT + "/acquire", json={
        "key": SECRET,
        "version": VERSION,
    })
    res.raise_for_status()
    res = res.json()

    command = ["../v2/daq"]
    command.extend(str(a) for a in res["args"])
    print("$", " ".join(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE, encoding="utf-8")
    while True:
        line = p.stdout.readline()
        if not line:
            break
        if line.startswith("SET "):
            _, k, v = line.split(" ")
            res[k] = int(v)

    res["key"] = SECRET
    res["version"] = VERSION
    print(res)
    requests.post(ENDPOINT + "/submit", json=res)
