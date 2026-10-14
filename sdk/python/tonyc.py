"""A Tonyc client in one file, using only the Python standard library.

It speaks the language-neutral ``tonyc/1`` protocol (docs/PROTOCOL.md) to a
running ``tonycd`` over its Unix socket::

    import tonyc
    t = tonyc.Client(origin="my-agent")
    hit = t.try_existing({"task": {"objective": "summarise the incidents"}})
    if hit["hit"]:
        print(hit["object"]["content"])
    else:
        job = t.submit({"task": {"objective": "summarise the incidents"},
                        "latency": {"deadline": "2h"}})
        print(t.wait(job["id"])["result"]["output"])

Requests are plain dicts shaped like the protocol's WorkRequest; answers are
the daemon's JSON, decoded.
"""

import http.client
import json
import os
import socket
import time
import urllib.parse

PROTOCOL = "tonyc/1"
TERMINAL = {
    "READY", "STALE", "CANCELLED", "DEADLINE_EXCEEDED", "POLICY_BLOCKED",
    "MODEL_FAILED", "NODE_UNREACHABLE", "INSUFFICIENT_QUALITY",
}


def default_socket():
    """$TONYC_SOCKET, else $XDG_RUNTIME_DIR/tonyc/tonycd.sock, else
    $XDG_STATE_HOME (default ~/.local/state)/tonyc/tonycd.sock."""
    for var in ("TONYC_SOCKET",):
        if os.environ.get(var):
            return os.environ[var]
    base = os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("XDG_STATE_HOME") \
        or os.path.join(os.environ.get("HOME", "/"), ".local/state")
    return os.path.join(base, "tonyc", "tonycd.sock")


class TonycError(Exception):
    """The daemon refused a request (``status``, ``message``, ``field``)."""

    def __init__(self, status, body):
        self.status = status
        self.body = body if isinstance(body, dict) else {"error": str(body)}
        self.message = self.body.get("error", "")
        self.field = self.body.get("field")
        super().__init__(f"tonycd answered {status}: {self.message}")


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("tonycd", timeout=timeout)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


class Client:
    """A client for one daemon. ``origin`` attributes submissions that carry
    none of their own to an orchestrator or agent system."""

    def __init__(self, socket_path=None, origin=None, agent=None):
        self.socket_path = socket_path or default_socket()
        self.origin = origin
        self.agent = agent

    # -------------------------------------------------------- transport --

    def _call(self, method, path, body=None, headers=None, timeout=660):
        conn = _UnixConnection(self.socket_path, timeout)
        try:
            data = None if body is None else json.dumps(body).encode()
            conn.request(method, path, body=data, headers=headers or {})
            res = conn.getresponse()
            raw = res.read()
        finally:
            conn.close()
        try:
            value = json.loads(raw) if raw else None
        except ValueError:
            value = raw.decode(errors="replace")
        if not 200 <= res.status < 300:
            raise TonycError(res.status, value)
        return res.status, value

    def _work(self, work):
        work = dict(work)
        if self.origin and not (work.get("origin") or {}).get("system"):
            work["origin"] = {"system": self.origin, "agent": self.agent}
        return work

    # --------------------------------------------------------- contract --

    def protocol(self):
        _, p = self._call("GET", "/v1/protocol")
        if p.get("protocol") != PROTOCOL:
            raise TonycError(0, {"error": f"tonycd speaks {p.get('protocol')}, this client {PROTOCOL}"})
        return p

    def submit(self, work, idempotency_key=None):
        """Queues work; returns ``{"id", "state", "envelope", "replayed"?}``."""
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return self._call("POST", "/v1/jobs", self._work(work), headers)[1]

    def status(self, job_id):
        return self._call("GET", f"/v1/jobs/{_q(job_id)}")[1]

    def await_job(self, job_id, timeout=30.0):
        """Long-polls up to ``timeout`` seconds; the job, terminal or not."""
        ms = int(min(timeout, 600) * 1000)
        return self._call("GET", f"/v1/jobs/{_q(job_id)}/await?timeout_ms={ms}")[1]

    def wait(self, job_id, timeout=3600.0):
        """Waits until the job is terminal; raises TimeoutError otherwise."""
        until = time.monotonic() + timeout
        while True:
            left = max(0.0, until - time.monotonic())
            job = self.await_job(job_id, min(left, 600))
            if job["job"]["state"] in TERMINAL:
                return job
            if left == 0:
                raise TimeoutError(f"job {job_id} did not finish in time")

    def cancel(self, job_id):
        return self._call("POST", f"/v1/jobs/{_q(job_id)}/cancel")[1]

    def try_existing(self, work):
        """Existing cognition now (``{"hit": true, "object": ...}``) or a miss."""
        return self._call("POST", "/v1/try", self._work(work))[1]

    def query(self, text=None, scope=None, kind=None, state=None, limit=None):
        params = {k: v for k, v in
                  dict(text=text, scope=scope, kind=kind, state=state, limit=limit).items() if v is not None}
        path = "/v1/cognition" + ("?" + urllib.parse.urlencode(params) if params else "")
        return self._call("GET", path)[1]["objects"]

    def invalidate(self, object=None, path=None, scope=None, event=None, reason=None):
        body = {k: v for k, v in
                dict(object=object, path=path, scope=scope, event=event, reason=reason).items() if v is not None}
        return self._call("POST", "/v1/invalidate", body)[1]

    def events(self, after=0, wait=0.0, system=None, limit=None):
        """One page of the event feed: ``{"events": [...], "next": cursor}``."""
        params = {"after": after, "wait_ms": int(min(wait, 60) * 1000)}
        if system:
            params["system"] = system
        if limit:
            params["limit"] = limit
        return self._call("GET", "/v1/events?" + urllib.parse.urlencode(params))[1]

    def follow(self, after=-1, system=None):
        """Yields feed events forever, resuming from the last cursor."""
        while True:
            page = self.events(after, wait=60, system=system)
            after = page["next"]
            yield from page["events"]


def _q(s):
    return urllib.parse.quote(s, safe="")
