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
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator

log = logging.getLogger("owoinfer")

PROTOCOL_VERSION = 1
ABI_VERSION = 1
CLIENT_NAME = "ComfyUI-MossImage-owoinfer"
SERVER_ENV = "MOSSINFER_SERVER"
SERVER_EXE = "moss-server.exe" if os.name == "nt" else "moss-server"
PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_NAME = "ComfyUI-MossImage-owoinfer.moss-server.log"

REQUEST_TIMEOUT = 120.0  # hello, load, status, cancel, unload: all answer quickly
SHUTDOWN_TIMEOUT = 5.0  # `shutdown` waits for a running job's cancel, then exits
CANCEL_SILENCE = 30.0  # after a cancel, this long without events or `status` answers means the server hangs
STATUS_EVERY = 5.0  # after a cancel, ask `status` this often while no events come (e.g. during a DiT load)
CANCEL_LIMIT = 600.0  # after a cancel, the most a job may take to end, however alive the server is
RELEASE_TIMEOUT = 600.0  # a release reads the DiT into RAM: seconds, at worst minutes
POLL = 0.1


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


def _adapter_stamp(path: str) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _is_exe(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _version_key(name: str) -> tuple:
    """Sort key for release folder names: the numbers in them, compared as numbers."""
    return tuple(int(n) for n in re.findall(r"\d+", name)), name


def pack_server_candidates(pack_dir: str = PACK_DIR) -> list[str]:
    """`<pack>/bin/moss-server`, or a release archive extracted into `<pack>/bin/`
    (`bin/bin/moss-server`, `bin/<archive folder>/bin/moss-server`, newest version first)."""
    bin_dir = os.path.join(pack_dir, "bin")
    found = [os.path.join(bin_dir, SERVER_EXE), os.path.join(bin_dir, "bin", SERVER_EXE)]
    try:
        subdirs = sorted(os.listdir(bin_dir), key=_version_key, reverse=True)
    except OSError:
        subdirs = []
    for name in subdirs:
        found.append(os.path.join(bin_dir, name, "bin", SERVER_EXE))
    return found


def locate_server(pack_dir: str = PACK_DIR, environ: dict[str, str] | None = None,
                  which: Callable[[str], str | None] = shutil.which) -> str:
    """The moss-server to run: $MOSSINFER_SERVER, then the pack's bin/ folder, then PATH.

    Never a value from a workflow: whoever controls the workflow would choose the program.
    """
    environ = os.environ if environ is None else environ
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
        self._log_start = self._log_file.tell()  # this process's lines start here
        self._ids = itertools.count(1)
        self._lock = threading.Lock()  # guards _pending, _subs, _dead_reason
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue] = {}
        self._subs: dict[str, queue.Queue] = {}
        self._dead_reason: str | None = None
        try:
            self.proc = _get_spawner().call(lambda: subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log_file,
                close_fds=True, env=env, **_popen_extra()))
        except BaseException:
            self._log_file.close()
            raise
        self._reader = threading.Thread(target=self._read_loop, name="moss-server-reader", daemon=True)
        self._reader.start()

    @property
    def alive(self) -> bool:
        return self._dead_reason is None and self.proc.poll() is None

    def next_id(self, prefix: str = "r") -> str:
        return f"{prefix}{next(self._ids)}"

    # reading -----------------------------------------------------------------

    def _read_loop(self) -> None:
        reason = "moss-server's output ended"
        try:
            stdout = self.proc.stdout
            assert stdout is not None
            for raw in iter(stdout.readline, b""):
                try:
                    line = raw.strip()
                    if line:
                        msg = json.loads(line)
                        if not isinstance(msg, dict):
                            raise ValueError("not a JSON object")
                        self._dispatch(msg)
                except Exception as e:  # one bad line must not stop the reader
                    log.warning("moss-server: bad protocol line (%s): %r", e, raw[:200])
            code = self.proc.wait()
            reason = f"moss-server exited with status {code}"
        except Exception as e:
            reason = f"reading moss-server's output failed: {e}"
        finally:
            # Every waiter wakes, whatever ended the loop.
            self._mark_dead(reason + self._log_tail())

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
            err = msg["error"] or {}
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
        """The last stderr lines of this process (not of earlier ones in the same file)."""
        try:
            if not self._log_file.closed:
                self._log_file.flush()
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(self._log_start, f.tell() - 4096))
                tail = f.read().decode("utf-8", "replace").splitlines()[-lines:]
        except (OSError, ValueError):
            return ""
        tail = [t for t in tail if t.strip()]
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

    def start_request(self, method: str, params: dict[str, Any] | None = None, *,
                      req_id: str | None = None) -> tuple[str, queue.Queue]:
        """Send a request; its response arrives on the returned queue (see reply())."""
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
        return req_id, slot

    @staticmethod
    def reply(msg: Any) -> dict[str, Any]:
        """A response's result; MossInferError for an error, ServerCrashed for a dead server."""
        if isinstance(msg, _Dead):
            raise ServerCrashed(msg.reason)
        if "error" in msg:
            err = msg["error"] or {}
            raise MossInferError(str(err.get("code", "internal")), str(err.get("message", "")))
        return msg.get("result") or {}

    def forget(self, req_id: str) -> None:
        with self._lock:
            self._pending.pop(req_id, None)

    def request(self, method: str, params: dict[str, Any] | None = None, *,
                req_id: str | None = None, timeout: float | None = REQUEST_TIMEOUT) -> dict[str, Any]:
        """Send a request and wait for its response; raise MossInferError on `error`."""
        req_id, slot = self.start_request(method, params, req_id=req_id)
        try:
            msg = slot.get(timeout=timeout)
        except queue.Empty:
            self.forget(req_id)
            raise MossInferError("timeout", f"moss-server did not answer {method} within {timeout:.0f} s") from None
        return self.reply(msg)

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

    def ensure_loaded(self, load_params: dict[str, Any], stamp: Any = None, adapters: Iterable[str] = ()) -> bool:
        """`load` unless the engine already runs exactly this config. True if it loaded.

        `stamp` (e.g. the model files' sizes and mtimes) is part of the config, so a
        replaced file loads again.

        `adapters` are the LoRA and control adapter files the next job uses. Engines up to
        1.0 (and no capability says which) key them by path, so one replaced under the same
        name would keep its old weights: when a path's (mtime, size) differs from the last
        time it ran on this engine, the engine is unloaded and loaded again first.
        """
        m = self._m
        key = json.dumps([load_params, stamp], sort_keys=True, default=str)
        stamps = {path: _adapter_stamp(path) for path in adapters}
        stale = [path for path, st in stamps.items() if path in m._adapter_stamps and m._adapter_stamps[path] != st]
        if key == m._loaded_key and not stale:
            m._adapter_stamps.update(stamps)
            return False
        if key == m._loaded_key and stale:
            log.info("moss-server: %s changed on disk; loading the engine again so it reads the new file",
                     ", ".join(os.path.basename(p) for p in stale))
            self.unload()
        m._loaded_key = None  # a failed load leaves no engine
        m._adapter_stamps = {}
        self.request("load", load_params)
        m._loaded_key = key
        m._adapter_stamps.update(stamps)
        return True

    def unload(self) -> None:
        self.request("unload")
        self._m._loaded_key = None
        self._m._adapter_stamps = {}

    def release(self, keep_host_copy: bool = True, *, should_cancel: Callable[[], bool] | None = None,
                timeout: float = RELEASE_TIMEOUT) -> dict[str, Any]:
        """`release` (a soft unload). It runs like a job: `should_cancel()` sends a `cancel`."""
        if not self.has("release"):
            raise MossInferError("not_supported",
                                 f"release needs a moss-server with the 'release' capability (this one is {self.version})")
        m = self._m
        proc = m._require_proc()
        try:
            req_id, slot = proc.start_request("release", {"keep_host_copy": keep_host_copy})
            deadline = time.monotonic() + timeout
            cancelled = False
            while True:
                try:
                    return proc.reply(slot.get(timeout=POLL))
                except queue.Empty:
                    pass
                if not cancelled and should_cancel is not None and should_cancel():
                    cancelled = True
                    proc.start_request("cancel", {"id": req_id})  # its answer is not needed
                if time.monotonic() > deadline:
                    proc.forget(req_id)
                    raise MossInferError("timeout", f"moss-server did not finish release within {timeout:.0f} s")
        except MossInferError as e:
            m._after_error(e)
            raise

    def run_job(self, method: str, params: dict[str, Any], *,
                on_event: Callable[[dict[str, Any]], None] | None = None,
                should_cancel: Callable[[], bool] | None = None,
                poll: float = POLL, cancel_silence: float = CANCEL_SILENCE,
                status_every: float = STATUS_EVERY, cancel_limit: float = CANCEL_LIMIT) -> dict[str, Any]:
        """Run a `generate` or `preload` and return its `done` event.

        Events other than `done` go to `on_event`. When `should_cancel()` turns true,
        or `on_event` raises, a `cancel` goes out and the job is drained to its `done`;
        an exception from `on_event` is then re-raised.

        After the cancel, phases without events (a first DiT load, the text encoder) can
        last a while, so a `status` request goes out every `status_every` seconds of
        silence, and its answers count as signs of life too. The process is killed when
        it shows none for `cancel_silence` seconds, or `cancel_limit` seconds after the cancel.
        """
        m = self._m
        proc = m._require_proc()
        job_id = proc.next_id("job")
        events = proc.subscribe(job_id)
        deferred: BaseException | None = None
        cancel_at: float | None = None
        last_life = 0.0
        status: tuple[str, queue.Queue] | None = None  # an unanswered `status` request
        try:
            try:
                proc.request(method, params, req_id=job_id)
            except MossInferError as e:
                m._after_error(e)
                raise
            while True:
                if cancel_at is None and (deferred is not None or (should_cancel is not None and should_cancel())):
                    try:
                        proc.start_request("cancel", {"id": job_id})  # its answer is not needed
                    except MossInferError as e:
                        m._after_error(e)
                        raise
                    cancel_at = last_life = time.monotonic()
                try:
                    msg = events.get(timeout=poll)
                except queue.Empty:
                    if cancel_at is None:
                        continue
                    now = time.monotonic()
                    if status is not None:
                        try:
                            answer = status[1].get_nowait()
                        except queue.Empty:
                            answer = None
                        if answer is not None:
                            status = None
                            if isinstance(answer, dict) and "result" in answer:
                                last_life = now
                    if status is None and now - last_life >= status_every:
                        try:
                            status = proc.start_request("status")
                        except MossInferError:
                            status = None  # the dead process shows up on `events` as well
                    if now - last_life > cancel_silence or now - cancel_at > cancel_limit:
                        m._discard_process(kill=True)
                        if deferred is not None:
                            raise deferred from None
                        raise MossInferError(
                            "cancel_timeout",
                            f"moss-server did not end the job within {now - cancel_at:.0f} s of a cancel; it was killed") from None
                    continue
                if cancel_at is not None:
                    last_life = time.monotonic()
                if isinstance(msg, _Dead):
                    m._discard_process()
                    if deferred is not None:
                        raise deferred
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
            if status is not None:
                proc.forget(status[0])
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
        self._adapter_stamps: dict[str, tuple[int, int] | None] = {}  # LoRA / control files used on this engine
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
        code = problem = ""
        if hello.get("protocol_version") != PROTOCOL_VERSION:
            code, problem = "unsupported_protocol", f"speaks protocol {hello.get('protocol_version')}, not {PROTOCOL_VERSION}"
        elif hello.get("abi_version") != ABI_VERSION:
            code, problem = "abi_mismatch", f"has engine ABI {hello.get('abi_version')}, not {ABI_VERSION}"
        if problem:
            self._discard_process()
            raise MossInferError(code, f"{server_path} {problem}; use a MossInfer 1.x release")
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
        self._adapter_stamps = {}
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
