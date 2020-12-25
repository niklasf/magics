import secrets
import aiohttp.web
import sqlite3
import sys

SECRET_KEY = sys.argv[1]
DEPTH = 2
STATS = 9
MIN_VERSION = 4

routes = aiohttp.web.RouteTableDef()

conn = sqlite3.connect("Bd5.db")


def cond(args):
    return " AND ".join(f"a{i} = {a:d}" for i, a in enumerate(args))

@routes.post("/acquire")
async def acquire(req: aiohttp.web.Request) -> aiohttp.web.Response:
    body = await req.json()
    key = body["key"]
    version = body["version"]

    if not secrets.compare_digest(key, SECRET_KEY):
        raise aiohttp.web.HTTPForbidden()

    with conn:
        fields = ", ".join([f"a{i}" for i in range(DEPTH)])
        args = conn.execute(f"SELECT {fields} FROM prefix WHERE acquired < {MIN_VERSION} LIMIT 1").fetchone()
        conn.execute(f"UPDATE prefix SET acquired = ? WHERE {cond(args)}", (version, ))

    return aiohttp.web.json_response({
        "args": args,
    })


@routes.post("/submit")
async def submit(req: aiohttp.web.Request) -> aiohttp.web.Response:
    body = await req.json()
    args = body["args"]
    key = body["key"]
    version = body["version"]
    magics = body["magics"]

    if not secrets.compare_digest(key, SECRET_KEY):
        raise aiohttp.web.HTTPForbidden()

    builder = []
    for i in range(STATS):
        name = f"s{i}"
        builder.append(f"{name} = {body[name]:d}")
    set_stats = ", ".join(builder)

    with conn:
        conn.execute(f"UPDATE prefix SET submitted = ?, magics = ?, {set_stats} WHERE {cond(args)} AND submitted < ?", (version, magics, version))

    raise aiohttp.web.HTTPNoContent()


@routes.get("/")
async def status(_req: aiohttp.web.Request) -> aiohttp.web.Response:
    with conn:
        total, = conn.execute("SELECT COUNT(*) FROM prefix").fetchone()
        submitted, = conn.execute(f"SELECT COUNT(*) FROM prefix WHERE submitted >= {MIN_VERSION}").fetchone()

    return aiohttp.web.Response(text=f"{submitted}/{total} = {submitted * 100 / total}")


def main() -> None:
    app = aiohttp.web.Application()
    app.add_routes(routes)
    aiohttp.web.run_app(app)


if __name__ == "__main__":
    main()
