"""The server manager against the fake moss-server: no GPU, no ComfyUI."""

import json
import os
import threading
import time

import pytest

from helpers import make_fake_server, recorded
from owoinfer_comfy import server as srv
from owoinfer_comfy.server import MossInferError, ServerCrashed, ServerManager, ServerNotFound


@pytest.fixture
def manager(tmp_path, fake_env):
    m = ServerManager(log_dir=str(tmp_path / "logs"), env=fake_env)
    yield m
    m.shutdown()


def gen_params(tmp_path, **kw):
    p = {"prompt": "a cat", "width": 64, "height": 48, "steps": 4, "seed": 3, "output": str(tmp_path / "out.png")}
    p.update(kw)
    return p


def load(session, tmp_path):
    model = tmp_path / "bundle"
    model.mkdir(exist_ok=True)
    return session.ensure_loaded({"model": str(model), "mode": "balanced"})


# locating -------------------------------------------------------------------------

def test_locate_order(tmp_path):
    pack = tmp_path / "pack"
    env_srv = make_fake_server(str(tmp_path / "envbin"))
    pack_srv = make_fake_server(str(pack / "bin"))
    path_srv = make_fake_server(str(tmp_path / "pathbin"))
    which = lambda name: path_srv  # noqa: E731
    assert srv.locate_server(pack_dir=str(pack), environ={srv.SERVER_ENV: env_srv}, which=which) == env_srv
    assert srv.locate_server(pack_dir=str(pack), environ={}, which=which) == pack_srv
    os.remove(pack_srv)
    assert srv.locate_server(pack_dir=str(pack), environ={}, which=which) == path_srv
    # An override wins over everything.
    assert srv.locate_server(env_srv, pack_dir=str(pack), environ={}, which=which) == env_srv


def test_locate_release_extracted_into_bin(tmp_path):
    pack = tmp_path / "pack"
    nested = make_fake_server(str(pack / "bin" / "mossinfer-1.0.0-linux-cu13-x86_64" / "bin"))
    assert srv.locate_server(pack_dir=str(pack), environ={}, which=lambda n: None) == nested


def test_locate_failure_names_every_place(tmp_path):
    with pytest.raises(ServerNotFound) as e:
        srv.locate_server(pack_dir=str(tmp_path), environ={}, which=lambda n: None)
    msg = str(e.value)
    assert srv.SERVER_ENV in msg and os.path.join(str(tmp_path), "bin") in msg and "PATH" in msg


def test_locate_bad_env_and_override(tmp_path):
    with pytest.raises(ServerNotFound, match=srv.SERVER_ENV):
        srv.locate_server(pack_dir=str(tmp_path), environ={srv.SERVER_ENV: str(tmp_path / "nope")})
    with pytest.raises(ServerNotFound, match="must name a moss-server"):
        srv.locate_server("/bin/sh", pack_dir=str(tmp_path), environ={})
    with pytest.raises(ServerNotFound, match="not an executable"):
        srv.locate_server(str(tmp_path / "moss-server"), pack_dir=str(tmp_path), environ={})


def test_probe_hello(fake_server, monkeypatch):
    monkeypatch.setenv("FAKE_MOSS_CAPS", "preview,release")
    hello = srv.probe_hello(fake_server)
    assert hello["protocol_version"] == 1
    assert hello["capabilities"]["release"] is True


# lifecycle --------------------------------------------------------------------------

def test_spawn_hello_and_parent_pid(manager, fake_server, fake_env):
    with manager.session(fake_server) as s:
        assert s.version == "1.0.0"
        assert s.has("control") and not s.has("release")
    argv = [p for m, p in recorded(fake_env) if m == "argv"][0]
    assert argv == ["--parent-pid", str(os.getpid())]
    hello = [p for m, p in recorded(fake_env) if m == "hello"][0]
    assert hello == {"client": srv.CLIENT_NAME, "protocol_version": 1}
    # stderr went to the log file
    with open(manager.log_path) as f:
        assert "fake moss-server 1.0.0 starting" in f.read()


def test_one_process_reused_and_load_only_on_change(manager, fake_server, fake_env, tmp_path):
    with manager.session(fake_server) as s:
        assert load(s, tmp_path) is True
        pid = manager.running.proc.pid
        s.run_job("generate", gen_params(tmp_path))
    with manager.session(fake_server) as s:
        assert load(s, tmp_path) is False  # same config: no second load
        assert manager.running.proc.pid == pid
        s.run_job("generate", gen_params(tmp_path))
        assert s.ensure_loaded({"model": str(tmp_path / "bundle"), "mode": "min_memory"}) is True
    methods = [m for m, _ in recorded(fake_env)]
    assert methods.count("hello") == 1 and methods.count("load") == 2 and methods.count("generate") == 2


def test_generate_events_and_output(manager, fake_server, tmp_path):
    seen = []
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        done = s.run_job("generate", gen_params(tmp_path, preview_output=str(tmp_path / "prev.png")),
                         on_event=seen.append)
    assert done["status"] == "ok" and done["image"] == str(tmp_path / "out.png")
    assert os.path.getsize(done["image"]) > 0
    kinds = [e["event"] for e in seen]
    assert kinds[0] == "plan" and "log" not in kinds
    assert [e["phase"] for e in seen if e["event"] == "progress"][:3] == ["text_encoder", "dit_load", "denoise"]
    assert sum(1 for k in kinds if k == "preview") == 4
    # each preview comes right before its step's progress
    i = kinds.index("preview")
    assert seen[i + 1]["event"] == "progress" and seen[i + 1]["step"] == seen[i]["step"]


def test_framing_unicode_and_newlines(manager, fake_server, fake_env, tmp_path):
    prompt = "猫\n\"quoted\" \\ back\tslash ünï"
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        s.run_job("generate", gen_params(tmp_path, prompt=prompt))
    sent = [p for m, p in recorded(fake_env) if m == "generate"][0]
    assert sent["prompt"] == prompt


def test_invalid_params_message_is_the_servers(manager, fake_server, tmp_path):
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        with pytest.raises(MossInferError) as e:
            s.run_job("generate", gen_params(tmp_path, width=1000))
    assert e.value.code == "invalid_params"
    assert str(e.value) == "Height and width must be positive multiples of 16."
    assert manager.running is not None  # a request error keeps the server


def test_load_error_leaves_no_engine(manager, fake_server, tmp_path):
    with manager.session(fake_server) as s:
        with pytest.raises(MossInferError) as e:
            s.ensure_loaded({"model": str(tmp_path / "missing")})
        assert e.value.code == "invalid_argument"
        with pytest.raises(MossInferError) as e:
            s.run_job("generate", gen_params(tmp_path))
        assert e.value.code == "no_engine"
        assert load(s, tmp_path) is True


def test_cancel(manager, fake_server, fake_env, tmp_path):
    fake_env["FAKE_MOSS_STEP_DELAY"] = "0.05"
    steps = []

    def on_event(e):
        if e["event"] == "progress" and e["phase"] == "denoise":
            steps.append(e["step"])

    with manager.session(fake_server) as s:
        load(s, tmp_path)
        done = s.run_job("generate", gen_params(tmp_path, steps=50), on_event=on_event,
                         should_cancel=lambda: len(steps) >= 3)
        assert done["status"] == "cancelled" and done["steps_run"] < 50
        # the engine stays usable after a cancel
        assert s.run_job("generate", gen_params(tmp_path, steps=2))["status"] == "ok"
    cancels = [p for m, p in recorded(fake_env) if m == "cancel"]
    gen_ids = [json.loads(line)["id"] for line in open(fake_env["FAKE_MOSS_RECORD"])
               if not line.startswith("argv") and json.loads(line).get("method") == "generate"]
    assert cancels == [{"id": gen_ids[0]}]


def test_exception_in_event_handler_cancels_then_reraises(manager, fake_server, fake_env, tmp_path):
    fake_env["FAKE_MOSS_STEP_DELAY"] = "0.05"

    class Interrupted(BaseException):
        pass

    def on_event(e):
        if e["event"] == "progress" and e["phase"] == "denoise" and e["step"] == 2:
            raise Interrupted()

    with manager.session(fake_server) as s:
        load(s, tmp_path)
        with pytest.raises(Interrupted):
            s.run_job("generate", gen_params(tmp_path, steps=50), on_event=on_event)
        status = s.request("status")
        assert status["state"] == "idle"  # drained to done before raising
    assert any(m == "cancel" for m, _ in recorded(fake_env))


def test_cancel_timeout_kills(manager, fake_server, fake_env, tmp_path):
    fake_env["FAKE_MOSS_STEP_DELAY"] = "0.2"
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        pid = manager.running.proc.pid
        with pytest.raises(MossInferError) as e:
            s.run_job("generate", gen_params(tmp_path, prompt="IGNORE_CANCEL", steps=100),
                      should_cancel=lambda: True, cancel_timeout=0.5)
    assert e.value.code == "cancel_timeout"
    assert manager.running is None
    with manager.session(fake_server) as s:
        assert manager.running.proc.pid != pid


def test_busy(manager, fake_server, fake_env, tmp_path):
    fake_env["FAKE_MOSS_STEP_DELAY"] = "0.05"
    errors = []

    def on_event(e):
        if e["event"] == "progress" and e["phase"] == "denoise" and e["step"] == 1 and not errors:
            # a second job sent straight to the process while one runs
            try:
                manager.running.request("generate", gen_params(tmp_path, output=str(tmp_path / "b.png")))
            except MossInferError as err:
                errors.append(err)

    with manager.session(fake_server) as s:
        load(s, tmp_path)
        assert s.run_job("generate", gen_params(tmp_path, steps=4), on_event=on_event)["status"] == "ok"
    assert errors and errors[0].code == "busy"


def test_session_lock_serializes_jobs(manager, fake_server, fake_env, tmp_path):
    fake_env["FAKE_MOSS_STEP_DELAY"] = "0.02"
    with manager.session(fake_server) as s:
        load(s, tmp_path)
    results, spans = [], []

    def worker(i):
        with manager.session(fake_server) as s:
            t0 = time.monotonic()
            results.append(s.run_job("generate", gen_params(tmp_path, output=str(tmp_path / f"o{i}.png")))["status"])
            spans.append((t0, time.monotonic()))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == ["ok"] * 3
    spans.sort()
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))  # never overlapping


def test_crash_then_restart(manager, fake_server, fake_env, tmp_path):
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        pid = manager.running.proc.pid
        with pytest.raises(ServerCrashed) as e:
            s.run_job("generate", gen_params(tmp_path, prompt="CRASH", steps=6))
    assert "status 134" in str(e.value) and "simulated fault" in str(e.value)
    assert manager.running is None
    with manager.session(fake_server) as s:
        assert manager.running.proc.pid != pid
        assert load(s, tmp_path) is True  # the new process needs its own load
        assert s.run_job("generate", gen_params(tmp_path))["status"] == "ok"
    with open(manager.log_path) as f:
        assert f.read().count("fake moss-server 1.0.0 starting") == 2


def test_cuda_error_restarts_process(manager, fake_server, tmp_path):
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        pid = manager.running.proc.pid
        done = s.run_job("generate", gen_params(tmp_path, prompt="CUDA_ERROR"))
    assert done["status"] == "error" and done["error"]["code"] == "cuda"
    assert manager.running is None
    with manager.session(fake_server) as s:
        assert manager.running.proc.pid != pid
        assert load(s, tmp_path) is True


def test_killed_while_idle_is_restarted(manager, fake_server, tmp_path):
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        proc = manager.running.proc
    proc.kill()
    proc.wait()
    time.sleep(0.1)
    with manager.session(fake_server) as s:
        assert manager.running.proc.pid != proc.pid
        assert load(s, tmp_path) is True


def test_capability_gating_release(tmp_path, fake_server, fake_env):
    old = ServerManager(log_dir=str(tmp_path / "a"), env=fake_env)
    try:
        with old.session(fake_server) as s:
            load(s, tmp_path)
            with pytest.raises(MossInferError) as e:
                s.release()
            assert e.value.code == "not_supported" and "'release'" in str(e.value)
    finally:
        old.shutdown()
    env = dict(fake_env, FAKE_MOSS_CAPS="preview,release,preload", FAKE_MOSS_VERSION="1.1.0")
    new = ServerManager(log_dir=str(tmp_path / "b"), env=env)
    try:
        with new.session(fake_server) as s:
            load(s, tmp_path)
            r = s.release(keep_host_copy=True)
            assert r["dit_host_copy"] is True and r["cancelled"] is False
            assert s.ensure_loaded({"model": str(tmp_path / "bundle"), "mode": "balanced"}) is False  # engine kept
            done = s.run_job("preload", {"width": 64, "height": 64, "steps": 4})
            assert done["status"] == "ok" and "dit_load" in done
    finally:
        new.shutdown()


def test_unload_then_load_again(manager, fake_server, fake_env, tmp_path):
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        s.unload()
        assert load(s, tmp_path) is True
    assert [m for m, _ in recorded(fake_env)].count("load") == 2


def test_protocol_mismatch(tmp_path, fake_server, fake_env):
    m = ServerManager(log_dir=str(tmp_path / "l"), env=dict(fake_env, FAKE_MOSS_PROTOCOL="2"))
    with pytest.raises(MossInferError) as e:
        with m.session(fake_server):
            pass
    assert e.value.code == "unsupported_protocol"
    assert m.running is None


def test_server_path_change_replaces_process(manager, fake_server, tmp_path):
    other = make_fake_server(str(tmp_path / "other"))
    with manager.session(fake_server):
        first = manager.running.proc
    with manager.session(other):
        assert manager.running.server_path == other
    assert first.poll() is not None


def test_shutdown_sends_shutdown_and_exits(manager, fake_server, fake_env, tmp_path):
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        proc = manager.running.proc
    manager.shutdown()
    assert proc.wait(timeout=5) == 0
    assert recorded(fake_env)[-1][0] == "shutdown"


def test_shutdown_during_job(manager, fake_server, fake_env, tmp_path):
    fake_env["FAKE_MOSS_STEP_DELAY"] = "0.05"
    result = {}

    def run():
        try:
            with manager.session(fake_server) as s:
                load(s, tmp_path)
                result["done"] = s.run_job("generate", gen_params(tmp_path, steps=100))
        except MossInferError as e:
            result["error"] = e

    t = threading.Thread(target=run)
    t.start()
    while manager.running is None:
        time.sleep(0.01)
    time.sleep(0.3)
    proc = manager.running.proc
    manager.shutdown()
    t.join(timeout=10)
    assert proc.poll() == 0
    # shutdown cancels the job first: its done arrives before the process exits
    assert result.get("done", {}).get("status") == "cancelled"


def test_progress_fraction_passthrough(tmp_path, fake_server, fake_env):
    env = dict(fake_env, FAKE_MOSS_CAPS="preview,progress_fraction")
    m = ServerManager(log_dir=str(tmp_path / "l"), env=env)
    seen = []
    try:
        with m.session(fake_server) as s:
            assert s.has("progress_fraction")
            load(s, tmp_path)
            s.run_job("generate", gen_params(tmp_path), on_event=seen.append)
    finally:
        m.shutdown()
    fractions = [e["fraction"] for e in seen if e["event"] == "progress"]
    assert fractions == sorted(fractions) and 0 <= fractions[0] and fractions[-1] <= 1
