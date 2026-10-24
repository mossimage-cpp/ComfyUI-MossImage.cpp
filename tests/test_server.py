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


def test_locate_takes_no_path_from_a_workflow():
    """No caller can choose the program: the only inputs are the environment, the pack and PATH."""
    import inspect
    assert list(inspect.signature(srv.locate_server).parameters) == ["pack_dir", "environ", "which"]


def test_locate_release_extracted_into_bin_newest_version_first(tmp_path):
    pack = tmp_path / "pack"
    make_fake_server(str(pack / "bin" / "mossinfer-1.9.0-linux-cu13-x86_64" / "bin"))
    newest = make_fake_server(str(pack / "bin" / "mossinfer-1.10.0-linux-cu13-x86_64" / "bin"))
    assert srv.locate_server(pack_dir=str(pack), environ={}, which=lambda n: None) == newest


def test_locate_failure_names_every_place(tmp_path):
    with pytest.raises(ServerNotFound) as e:
        srv.locate_server(pack_dir=str(tmp_path), environ={}, which=lambda n: None)
    msg = str(e.value)
    assert srv.SERVER_ENV in msg and os.path.join(str(tmp_path), "bin") in msg and "PATH" in msg


def test_locate_bad_env(tmp_path):
    with pytest.raises(ServerNotFound, match=srv.SERVER_ENV):
        srv.locate_server(pack_dir=str(tmp_path), environ={srv.SERVER_ENV: str(tmp_path / "nope")})


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


def test_silent_server_after_cancel_is_killed(manager, fake_server, fake_env, tmp_path):
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        pid = manager.running.proc.pid
        with pytest.raises(MossInferError) as e:
            s.run_job("generate", gen_params(tmp_path, prompt="HANG_ON_CANCEL", steps=100),
                      should_cancel=lambda: True, cancel_silence=0.5, status_every=0.1)
    assert e.value.code == "cancel_timeout"  # `status` went unanswered too
    assert manager.running is None
    with manager.session(fake_server) as s:
        assert manager.running.proc.pid != pid


def test_status_answers_keep_a_silent_job_alive(manager, fake_server, fake_env, tmp_path):
    """No events for longer than the silence limit (a DiT load, say), but `status` is answered."""
    fake_env["FAKE_MOSS_SLOW_CANCEL"] = "1.5"
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        pid = manager.running.proc.pid
        done = s.run_job("generate", gen_params(tmp_path, prompt="SLOW_CANCEL", steps=100),
                         should_cancel=lambda: True, cancel_silence=0.6, status_every=0.2)
        assert done["status"] == "cancelled" and manager.running.proc.pid == pid
    assert [m for m, _ in recorded(fake_env)].count("status") >= 3


def test_cancel_limit(manager, fake_server, fake_env, tmp_path):
    """However alive the server is, a job that does not end long after its cancel is killed."""
    fake_env["FAKE_MOSS_SLOW_CANCEL"] = "30"
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        t0 = time.monotonic()
        with pytest.raises(MossInferError) as e:
            s.run_job("generate", gen_params(tmp_path, prompt="SLOW_CANCEL", steps=100),
                      should_cancel=lambda: True, cancel_silence=0.6, status_every=0.2, cancel_limit=1.0)
    assert e.value.code == "cancel_timeout" and time.monotonic() - t0 < 5
    assert manager.running is None


def test_server_still_talking_after_cancel_is_not_killed(manager, fake_server, fake_env, tmp_path):
    """The silence limit restarts with every event: a job that keeps reporting is waited for."""
    fake_env["FAKE_MOSS_STEP_DELAY"] = "0.05"
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        pid = manager.running.proc.pid
        done = s.run_job("generate", gen_params(tmp_path, prompt="IGNORE_CANCEL", steps=20),
                         should_cancel=lambda: True, cancel_silence=0.3)  # the job lasts about 1 s
        assert done["status"] == "ok" and manager.running.proc.pid == pid


def test_interrupt_wins_over_cancel_timeout(manager, fake_server, tmp_path):
    class Interrupted(BaseException):
        pass

    def on_event(e):
        if e["event"] == "progress" and e["phase"] == "denoise":
            raise Interrupted()

    with manager.session(fake_server) as s:
        load(s, tmp_path)
        with pytest.raises(Interrupted):
            s.run_job("generate", gen_params(tmp_path, prompt="HANG_ON_CANCEL", steps=100),
                      on_event=on_event, cancel_silence=0.5)
    assert manager.running is None  # killed all the same


def test_bad_lines_do_not_stop_the_reader(manager, fake_server, tmp_path):
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        assert s.run_job("generate", gen_params(tmp_path, prompt="BAD_LINES"))["status"] == "ok"
        assert s.request("status")["state"] == "idle"


def test_busy_from_the_server(manager, fake_server, fake_env, tmp_path):
    """A `busy` answer (another client of the same process holds a job) is an error for this
    job only: the process stays, and the next job runs once the other ends."""
    fake_env["FAKE_MOSS_STEP_DELAY"] = "0.05"
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        proc = manager.running
        other = proc.next_id("other")
        proc.request("generate", gen_params(tmp_path, steps=6, output=str(tmp_path / "other.png")), req_id=other)
        with pytest.raises(MossInferError) as e:
            s.run_job("generate", gen_params(tmp_path))
        assert e.value.code == "busy"
        assert manager.running is proc
        while s.request("status")["state"] != "idle":
            time.sleep(0.02)
        assert s.run_job("generate", gen_params(tmp_path))["status"] == "ok"


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


@pytest.fixture
def release_manager(tmp_path, fake_env):
    env = dict(fake_env, FAKE_MOSS_CAPS="preview,release", FAKE_MOSS_RELEASE_DELAY="5")
    m = ServerManager(log_dir=str(tmp_path / "r"), env=env)
    yield m
    m.shutdown()


def test_release_can_be_cancelled(release_manager, fake_server, tmp_path):
    with release_manager.session(fake_server) as s:
        load(s, tmp_path)
        t0 = time.monotonic()
        r = s.release(should_cancel=lambda: time.monotonic() - t0 > 0.2)
        assert r["cancelled"] is True and r["vram_released_bytes"] == 0
        assert time.monotonic() - t0 < 3
        assert s.request("status")["state"] == "idle"


def test_release_has_a_finite_timeout(release_manager, fake_server, tmp_path):
    with release_manager.session(fake_server) as s:
        load(s, tmp_path)
        with pytest.raises(MossInferError) as e:
            s.release(timeout=0.3)
    assert e.value.code == "timeout"
    assert release_manager.running is None  # a server that does not answer is killed


def test_replaced_model_file_loads_again(manager, fake_server, fake_env, tmp_path):
    params = {"model": str(tmp_path / "bundle"), "mode": "balanced"}
    (tmp_path / "bundle").mkdir()
    with manager.session(fake_server) as s:
        assert s.ensure_loaded(params, stamp=[("/d.safetensors", 1, 10)]) is True
        assert s.ensure_loaded(params, stamp=[("/d.safetensors", 1, 10)]) is False
        assert s.ensure_loaded(params, stamp=[("/d.safetensors", 2, 10)]) is True
    assert [m for m, _ in recorded(fake_env)].count("load") == 2


def test_unload_then_load_again(manager, fake_server, fake_env, tmp_path):
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        s.unload()
        assert load(s, tmp_path) is True
    assert [m for m, _ in recorded(fake_env)].count("load") == 2


def test_replaced_lora_reloads_the_engine(manager, fake_server, fake_env, tmp_path):
    """Engines up to 1.0 key LoRAs and control adapters by path: a file replaced under the
    same name needs a new engine (unload, load) before the next generate."""
    lora = tmp_path / "style.safetensors"
    adapter = tmp_path / "control.safetensors"
    lora.write_bytes(b"one")
    adapter.write_bytes(b"adapter")
    params = {"model": str(tmp_path / "bundle"), "mode": "balanced"}
    (tmp_path / "bundle").mkdir()

    def run(s):
        s.ensure_loaded(params, adapters=[str(lora), str(adapter)])
        s.run_job("generate", gen_params(tmp_path, loras=[{"path": str(lora), "strength": 1.0}]))

    def methods_since(n):
        return [m for m, _ in recorded(fake_env)][n:]

    with manager.session(fake_server) as s:
        run(s)
        n = len(recorded(fake_env))
        run(s)  # unchanged: no reload
        assert methods_since(n) == ["generate"]
        n = len(recorded(fake_env))
        lora.write_bytes(b"three")  # replaced under the same name
        run(s)
        assert methods_since(n) == ["unload", "load", "generate"]
        n = len(recorded(fake_env))
        run(s)  # the new stamp is recorded
        assert methods_since(n) == ["generate"]
        n = len(recorded(fake_env))
        adapter.write_bytes(b"new adapter")
        run(s)
        assert methods_since(n) == ["unload", "load", "generate"]
        # a load for another reason clears the record: nothing stale afterwards
        assert s.ensure_loaded(dict(params, mode="min_memory"), adapters=[str(lora)]) is True
        n = len(recorded(fake_env))
        assert s.ensure_loaded(dict(params, mode="min_memory"), adapters=[str(lora), str(adapter)]) is False
        assert methods_since(n) == []


@pytest.mark.parametrize("var,value,code,word", [("FAKE_MOSS_PROTOCOL", "2", "unsupported_protocol", "unsupported protocol"),
                                                  ("FAKE_MOSS_ABI", "2", "abi_mismatch", "ABI 2")])
def test_protocol_or_abi_mismatch(tmp_path, fake_server, fake_env, var, value, code, word):
    m = ServerManager(log_dir=str(tmp_path / "l"), env=dict(fake_env, **{var: value}))
    with pytest.raises(MossInferError) as e:
        with m.session(fake_server):
            pass
    assert e.value.code == code and word in str(e.value)
    assert m.running is None


def test_log_tail_quotes_only_the_current_process(manager, fake_server, tmp_path):
    os.makedirs(manager.log_dir)
    with open(manager.log_path, "w") as f:
        f.write("an old line from an earlier process\n")
    manager._log_rotated = True  # keep the old line in the same file
    with manager.session(fake_server) as s:
        load(s, tmp_path)
        with pytest.raises(ServerCrashed) as e:
            s.run_job("generate", gen_params(tmp_path, prompt="CRASH", steps=6))
    assert "simulated fault" in str(e.value) and "earlier process" not in str(e.value)


def test_spawn_failure_closes_the_log(tmp_path, monkeypatch):
    opened = []
    real_open = open

    def tracking_open(*a, **kw):
        f = real_open(*a, **kw)
        opened.append(f)
        return f

    monkeypatch.setattr("builtins.open", tracking_open)
    bad = tmp_path / "moss-server"
    bad.write_text("not a program")
    bad.chmod(0o755)
    m = ServerManager(log_dir=str(tmp_path / "l"))
    with pytest.raises(ServerNotFound):
        with m.session(str(bad)):
            pass
    assert opened and all(f.closed for f in opened)


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
