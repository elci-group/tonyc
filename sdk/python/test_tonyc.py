"""Exercises sdk/python/tonyc.py against a running daemon (TONYC_SOCKET).
Run by scripts/e2e.sh with the built-in runtime."""

import sys
import tonyc

t = tonyc.Client(origin="python-test")
p = t.protocol()
assert "events" in p["features"], p

# The floor suits the built-in echo runtime the e2e suite runs on.
work = {"task": {"objective": "python says hello", "input": {"text": "x y z"}},
        "quality": {"floor": 0.2}, "reuse": {"corpus": "project:py"}}
job = t.submit(work, idempotency_key="py-1")
again = t.submit(work, idempotency_key="py-1")
assert again["id"] == job["id"] and again.get("replayed"), again
try:
    t.submit({"task": {"objective": "something else"}}, idempotency_key="py-1")
    raise AssertionError("conflict expected")
except tonyc.TonycError as e:
    assert e.status == 409, e

done = t.wait(job["id"], timeout=30)
assert done["job"]["state"] == "READY", done
assert done["job"]["envelope"]["origin"]["system"] == "python-test"
assert "python says hello" in done["result"]["output"]

hit = t.try_existing(work)
assert hit["hit"], hit
assert t.query(text="hello", scope="project:py")[0]["id"] == hit["object"]["id"]

page = t.events(after=0, system="python-test")
assert any(e["job_id"] == job["id"] and e["terminal"] for e in page["events"]), page

report = t.invalidate(scope="project:py", reason="python test")
assert job["id"] in report["jobs"], report
assert not t.try_existing(work)["hit"]

try:
    t.submit({"task": {"objective": "x"}, "quality": {"floor": 7}})
    raise AssertionError("refusal expected")
except tonyc.TonycError as e:
    assert e.status == 400 and e.field == "quality.floor", e

print("python client: all checks passed")
sys.exit(0)
