"""moss-server sidecar client: locating, spawning, framing, one job at a time.

Standard library only, with no ComfyUI imports, so it can be tested on its own.
The protocol is MossInfer's docs/server-protocol.md (protocol 1).
"""

from __future__ import annotations

import atexit
import itertools
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

log = logging.getLogger("owoinfer")

PROTOCOL_VERSION = 1
CLIENT_NAME = "ComfyUI-MossImage-owoinfer"
SERVER_ENV = "MOSSINFER_SERVER"
SERVER_EXE = "moss-server.exe" if os.name == "nt" else "moss-server"
PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_NAME = "ComfyUI-MossImage-owoinfer.moss-server.log"

REQUEST_TIMEOUT = 120.0  # hello, load, status, cancel, unload: all answer quickly
SHUTDOWN_TIMEOUT = 5.0  # `shutdown` waits for a running job's cancel, then exits
CANCEL_TIMEOUT = 30.0  # a cancel stops a job within one layer or block


class MossInferError(Exception):
    """An error from moss-server ({code, message}) or from talking to it.

    str(error) is the server's message as it is, so ComfyUI shows it unchanged.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ServerNotFound(MossInferError):
    def __init__(self, message: str):
        super().__init__("server_not_found", message)


class ServerCrashed(MossInferError):
    def __init__(self, message: str):
        super().__init__("crashed", message)


def _is_exe(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def pack_server_candidates(pack_dir: str = PACK_DIR) -> list[str]:
    """`<pack>/bin/moss-server`, or a release archive extracted into `<pack>/bin/`
    (`bin/bin/moss-server`, `bin/<archive folder>/bin/moss-server`)."""
    bin_dir = os.path.join(pack_dir, "bin")
    found = [os.path.join(bin_dir, SERVER_EXE), os.path.join(bin_dir, "bin", SERVER_EXE)]
    try:
        subdirs = sorted(os.listdir(bin_dir), reverse=True)  # newest version name first
    except OSError:
        subdirs = []
    for name in subdirs:
        found.append(os.path.join(bin_dir, name, "bin", SERVER_EXE))
    return found


def locate_server(
    override: str = "",
    pack_dir: str = PACK_DIR,
    environ: dict[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    """The moss-server to run: `override`, then $MOSSINFER_SERVER, then the pack's
    bin/ folder, then PATH."""
    environ = os.environ if environ is None else environ
    override = override.strip()
    if override:
        if os.path.basename(override).lower() != SERVER_EXE:
            # A workflow file carries this value; never run another program from it.
            raise ServerNotFound(f"server path must name a {SERVER_EXE} binary: {override}")
        if not _is_exe(override):
            raise ServerNotFound(f"server path is not an executable file: {override}")
        return os.path.abspath(override)
    env_path = environ.get(SERVER_ENV, "").strip()
    if env_path:
        if not _is_exe(env_path):
            raise ServerNotFound(f"{SERVER_ENV} is set but is not an executable file: {env_path}")
        return os.path.abspath(env_path)
    for candidate in pack_server_candidates(pack_dir):
        if _is_exe(candidate):
            return candidate
    on_path = which(SERVER_EXE)
    if on_path:
        return os.path.abspath(on_path)
    raise ServerNotFound(
        f"{SERVER_EXE} not found. Looked at: the {SERVER_ENV} environment variable (unset); "
        f"{os.path.join(pack_dir, 'bin', SERVER_EXE)} (extract a MossInfer release into "
        f"{os.path.join(pack_dir, 'bin')}); and PATH."
    )


def probe_hello(server_path: str, timeout: float = 20.0) -> dict[str, Any]:
    """A one-shot `hello` from a short-lived server process (no engine, no VRAM)."""
    lines = (
        json.dumps({"id": "probe", "method": "hello",
                    "params": {"client": CLIENT_NAME, "protocol_version": PROTOCOL_VERSION}})
        + "\n" + json.dumps({"id": "bye", "method": "shutdown"}) + "\n"
    )
    proc = subprocess.run([server_path], input=lines.encode(), stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, timeout=timeout, **_popen_extra())
    for raw in proc.stdout.splitlines():
        try:
            msg = json.loads(raw)
        except ValueError:
            continue
        if msg.get("id") == "probe":
            if "error" in msg:
                err = msg["error"]
                raise MossInferError(err.get("code", "internal"), err.get("message", ""))
            return msg["result"]
    raise ServerCrashed(f"{server_path} exited (status {proc.returncode}) without answering hello")


def _popen_extra() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    # Own session: a Ctrl-C in ComfyUI's terminal must not kill the server before
    # the atexit `shutdown`; --parent-pid still ends it if ComfyUI dies.
    return {"start_new_session": True}


class _Spawner:
    """A thread that lives as long as the interpreter and spawns every server.

    On Linux, moss-server --parent-pid sets PR_SET_PDEATHSIG, which fires when
    the spawning *thread* exits, so spawning from a short-lived thread would kill
    the server early (server-protocol.md, "--parent-pid").
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="moss-server-spawner", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            fn, box, done = self._jobs.get()
            try:
                box.append((True, fn()))
            except BaseException as e:  # handed back to the caller
                box.append((False, e))
            done.set()

    def call(self, fn: Callable[[], Any]) -> Any:
        box: list = []
        done = threading.Event()
        self._jobs.put((fn, box, done))
        done.wait()
        ok, value = box[0]
        if not ok:
            raise value
        return value


_spawner: _Spawner | None = None
_spawner_lock = threading.Lock()


def _get_spawner() -> _Spawner:
    global _spawner
    with _spawner_lock:
        if _spawner is None:
            _spawner = _Spawner()
        return _spawner


class _Dead:
    """Queued to every waiter when the server process ends."""

    def __init__(self, reason: str):
        self.reason = reason


class ServerProcess:
    """One moss-server process: pipes, a reader thread, responses matched by id."""

    def __init__(self, server_path: str, log_path: str, parent_pid: int | None = None,
                 env: dict[str, str] | None = None):
        self.server_path = server_path
        self.log_path = log_path
        argv = [server_path]
        if parent_pid is not None:
            argv += ["--parent-pid", str(parent_pid)]
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._log_file = open(log_path, "ab")
        self._log_file.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n".encode())
        self._log_file.flush()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()  # guards _pending, _subs, _dead_reason
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._subs: dict[str, queue.Queue] = {}
        self._dead_reason: str | None = None
        self.proc = _get_spawner().call(lambda: subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log_file,
            bufsize=0, close_fds=True, env=env, **_popen_extra()))
        self._reader = threading.Thread(target=self._read_loop, name="moss-server-reader", daemon=True)
        self._reader.start()

    @property
    def alive(self) -> bool:
        return self._dead_reason is None and self.proc.poll() is None

    def next_id(self, prefix: str = "r") -> str:
        return f"{prefix}{next(self._ids)}"

    # reading -----------------------------------------------------------------

    def _read_loop(self) -> None:
        stdout = self.proc.stdout
        assert stdout is not None
        for raw in iter(stdout.readline, b""):
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                log.warning("moss-server: unparsable line: %r", line[:200])
                continue
            if isinstance(msg, dict):
                self._dispatch(msg)
        code = self.proc.wait()
        self._mark_dead(f"moss-server exited with status {code}{self._log_tail()}")

    def _dispatch(self, msg: dict[str, Any]) -> None:
        msg_id = msg.get("id")
        key = msg_id if isinstance(msg_id, str) else None
        if "event" in msg:
            if msg["event"] == "log":
                level = logging.WARNING if msg.get("level") == "warning" else logging.INFO
                log.log(level, "moss-server: %s", msg.get("message", ""))
                return
            with self._lock:
                sub = self._subs.get(key) if key is not None else None
            if sub is not None:
                sub.put(msg)
            return
        with self._lock:
            slot = self._pending.pop(key, None) if key is not None else None
        if slot is not None:
            slot.put(msg)
        elif "error" in msg:
            err = msg["error"]
            log.error("moss-server: %s: %s", err.get("code"), err.get("message"))

    def _mark_dead(self, reason: str) -> None:
        with self._lock:
            if self._dead_reason is not None:
                return
            self._dead_reason = reason
            waiters = list(self._pending.values()) + list(self._subs.values())
            self._pending.clear()
        for w in waiters:
            w.put(_Dead(reason))
        try:
            self._log_file.close()
        except OSError:
            pass

    def _log_tail(self, lines: int = 6) -> str:
        try:
            self._log_file.flush()
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", "replace").splitlines()[-lines:]
        except (OSError, ValueError):
            return ""
        tail = [t for t in tail if t.strip() and not t.startswith("=== ")]
        return (". Last log lines: " + " | ".join(tail)) if tail else f". Log: {self.log_path}"

    # writing -----------------------------------------------------------------

    def _send(self, msg: dict[str, Any]) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as e:
                self._mark_dead(f"moss-server is not running ({e}){self._log_tail()}")
                raise ServerCrashed(self._dead_reason or "moss-server is not running") from None

    def request(self, method: str, params: dict[str, Any] | None = None, *,
                req_id: str | None = None, timeout: float | None = REQUEST_TIMEOUT) -> dict[str, Any]:
        """Send a request and wait for its response; raise MossInferError on `error`."""
        req_id = req_id or self.next_id()
        slot: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            if self._dead_reason is not None:
                raise ServerCrashed(self._dead_reason)
            self._pending[req_id] = slot
        msg: dict[str, Any] = {"id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        try:
            reply = slot.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                self._pending.pop(req_id, None)
            raise MossInferError("timeout", f"moss-server did not answer {method} within {timeout:.0f} s") from None
        if isinstance(reply, _Dead):
            raise ServerCrashed(reply.reason)
        if "error" in reply:
            err = reply["error"] or {}
            raise MossInferError(str(err.get("code", "internal")), str(err.get("message", "")))
        return reply.get("result") or {}

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            if self._dead_reason is not None:
                q.put(_Dead(self._dead_reason))
            self._subs[job_id] = q
        return q

    def unsubscribe(self, job_id: str) -> None:
        with self._lock:
            self._subs.pop(job_id, None)

    # stopping ----------------------------------------------------------------

    def stop(self, timeout: float = SHUTDOWN_TIMEOUT) -> None:
        """`shutdown`, then kill if the process has not exited after `timeout`."""
        if self.proc.poll() is None:
            try:
                self.request("shutdown", timeout=timeout)
            except MossInferError:
                pass
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.warning("moss-server did not exit after shutdown; killing it")
                self.kill()
        self._reader.join(timeout=timeout)

    def kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=SHUTDOWN_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired):
            pass


class Session:
    """The manager while its job lock is held: see ServerManager.session()."""

    def __init__(self, manager: "ServerManager"):
        self._m = manager

    @property
    def hello(self) -> dict[str, Any]:
        return self._m._hello or {}

    @property
    def capabilities(self) -> dict[str, bool]:
        return dict(self.hello.get("capabilities") or {})

    @property
    def version(self) -> str:
        return str(self.hello.get("version", "?"))

    def has(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability))

    def request(self, method: str, params: dict[str, Any] | None = None,
                timeout: float | None = REQUEST_TIMEOUT) -> dict[str, Any]:
        proc = self._m._require_proc()
        try:
            return proc.request(method, params, timeout=timeout)
        except MossInferError as e:
            self._m._after_error(e)
            raise

    def ensure_loaded(self, load_params: dict[str, Any]) -> bool:
        """`load` unless the engine already runs exactly this config. True if it loaded."""
        key = json.dumps(load_params, sort_keys=True)
        if key == self._m._loaded_key:
            return False
        self._m._loaded_key = None  # a failed load leaves no engine
        self.request("load", load_params)
        self._m._loaded_key = key
        return True

    def unload(self) -> None:
        self.request("unload")
        self._m._loaded_key = None

    def release(self, keep_host_copy: bool = True) -> dict[str, Any]:
        if not self.has("release"):
            raise MossInferError("not_supported",
                                 f"release needs a moss-server with the 'release' capability (this one is {self.version})")
        return self.request("release", {"keep_host_copy": keep_host_copy}, timeout=None)

    def run_job(self, method: str, params: dict[str, Any], *,
                on_event: Callable[[dict[str, Any]], None] | None = None,
                should_cancel: Callable[[], bool] | None = None,
                poll: float = 0.1, cancel_timeout: float = CANCEL_TIMEOUT) -> dict[str, Any]:
        """Run a `generate` or `preload` and return its `done` event.

        Events other than `done` go to `on_event`. When `should_cancel()` turns true,
        or `on_event` raises, a `cancel` goes out and the job is drained to its `done`;
        an exception from `on_event` is then re-raised. A job that does not end within
        `cancel_timeout` after the cancel has its process killed.
        """
        m = self._m
        proc = m._require_proc()
        job_id = proc.next_id("job")
        events = proc.subscribe(job_id)
        deferred: BaseException | None = None
        cancel_at: float | None = None
        try:
            try:
                proc.request(method, params, req_id=job_id)
            except MossInferError as e:
                m._after_error(e)
                raise
            while True:
                if cancel_at is None and (deferred is not None or (should_cancel is not None and should_cancel())):
                    cancel_at = time.monotonic()
                    try:
                        proc.request("cancel", {"id": job_id})
                    except MossInferError as e:
                        m._after_error(e)
                        raise
                if cancel_at is not None and time.monotonic() - cancel_at > cancel_timeout:
                    m._discard_process(kill=True)
                    raise MossInferError("cancel_timeout",
                                         f"moss-server did not stop within {cancel_timeout:.0f} s of a cancel; it was killed")
                try:
                    msg = events.get(timeout=poll)
                except queue.Empty:
                    continue
                if isinstance(msg, _Dead):
                    m._discard_process()
                    raise ServerCrashed(msg.reason)
                if msg.get("event") == "done":
                    done = msg
                    break
                if on_event is not None and deferred is None:
                    try:
                        on_event(msg)
                    except BaseException as e:  # re-raised once the server has stopped
                        deferred = e
        finally:
            proc.unsubscribe(job_id)
        if done.get("status") == "cancelled":
            log.info("moss-server: %s cancelled after %s steps", method, done.get("steps_run"))
        if deferred is not None:
            raise deferred
        err = done.get("error") or {}
        if done.get("status") == "error" and err.get("code") == "cuda":
            log.warning("moss-server reported a CUDA error; restarting it for the next run")
            m._discard_process()
        return done


class ServerManager:
    """One moss-server per ComfyUI process, reused across runs; one job at a time."""

    def __init__(self, log_dir: str, cache_dir: str | None = None,
                 parent_pid: int | None = None, env: dict[str, str] | None = None):
        self.log_dir = log_dir
        self.cache_dir = cache_dir
        self.parent_pid = os.getpid() if parent_pid is None else parent_pid
        self.env = env
        self._lock = threading.Lock()
        self._proc: ServerProcess | None = None
        self._hello: dict[str, Any] | None = None
        self._loaded_key: str | None = None
        self._log_rotated = False
        atexit.register(self.shutdown)

    @property
    def log_path(self) -> str:
        return os.path.join(self.log_dir, LOG_NAME)

    @property
    def running(self) -> ServerProcess | None:
        proc = self._proc
        return proc if proc is not None and proc.alive else None

    @contextmanager
    def session(self, server_path: str) -> Iterator[Session]:
        """Hold the job lock, with a live, hello'd server for `server_path`."""
        with self._lock:
            self._ensure_process(server_path)
            yield Session(self)

    def cached_hello(self, server_path: str) -> dict[str, Any] | None:
        proc = self.running
        if proc is not None and proc.server_path == server_path:
            return self._hello
        return None

    def _ensure_process(self, server_path: str) -> None:
        if self._proc is not None and (not self._proc.alive or self._proc.server_path != server_path):
            self._discard_process()
        if self._proc is not None:
            return
        if not self._log_rotated:
            self._log_rotated = True
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                if os.path.exists(self.log_path):
                    os.replace(self.log_path, self.log_path + ".1")
            except OSError:
                pass
        try:
            proc = ServerProcess(server_path, self.log_path, self.parent_pid, self.env)
        except OSError as e:
            raise ServerNotFound(f"could not start {server_path}: {e}") from None
        self._proc = proc
        try:
            hello = proc.request("hello", {"client": CLIENT_NAME, "protocol_version": PROTOCOL_VERSION})
        except MossInferError:
            self._discard_process(kill=True)
            raise
        if hello.get("protocol_version") != PROTOCOL_VERSION:
            self._discard_process()
            raise MossInferError("unsupported_protocol",
                                 f"{server_path} speaks protocol {hello.get('protocol_version')}, not {PROTOCOL_VERSION}")
        self._hello = hello
        probe = hello.get("probe") or {}
        log.info("moss-server %s started (%s); %s", hello.get("version"), server_path, probe.get("message", ""))

    def _require_proc(self) -> ServerProcess:
        if self._proc is None:
            raise ServerCrashed("moss-server is not running")
        return self._proc

    def _after_error(self, e: MossInferError) -> None:
        if isinstance(e, ServerCrashed) or e.code == "timeout":
            self._discard_process(kill=True)
        elif e.code == "cuda":
            log.warning("moss-server reported a CUDA error; restarting it for the next run")
            self._discard_process()

    def _discard_process(self, kill: bool = False) -> None:
        proc, self._proc = self._proc, None
        self._hello = None
        self._loaded_key = None
        if proc is None:
            return
        if kill:
            proc.kill()
        else:
            proc.stop()

    def shutdown(self) -> None:
        """At exit: `shutdown` (which cancels a running job), kill after a timeout.

        Does not take the job lock: a generation may hold it in ComfyUI's worker thread.
        """
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.stop()


if __name__ == "__main__":  # `python -m owoinfer_comfy.server [path]`: print hello
    logging.basicConfig(level=logging.INFO)
    path = locate_server(sys.argv[1] if len(sys.argv) > 1 else "")
    print(json.dumps(probe_hello(path), indent=2))
