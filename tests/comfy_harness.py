"""Headless ComfyUI for the e2e tests: the user's install, run read-only.

ComfyUI runs from its own code and venv with a scratch --base-directory whose
custom_nodes/ holds a symlink to this pack, and --cpu so its torch takes no VRAM.
Nothing is written into the install: bytecode writing is off, and the database
(by default <install>/user/comfyui.db, whatever --base-directory says) lives in
the scratch directory. install_changes() checks that.

ComfyUI always reads <install>/extra_model_paths.yaml when that file exists;
there is no option to skip it. It is only read, but its folders then show up in
the tests' ComfyUI too. install_extra_paths() names it so a fixture can warn.
"""

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

from helpers import ROOT

COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "/workspace/AI/ComfyUI")
COMFYUI_PYTHON = os.environ.get("COMFYUI_PYTHON", os.path.join(COMFYUI_DIR, ".venv", "bin", "python"))
PACK_NAME = "ComfyUI-MossImage.cpp"
WORKFLOW = os.path.join(ROOT, "example_workflows", "MOSS-Image_txt2img.json")


def comfyui_missing() -> str:
    """Why ComfyUI cannot run here, or ""."""
    if not os.path.isfile(os.path.join(COMFYUI_DIR, "main.py")):
        return f"no ComfyUI at {COMFYUI_DIR} (set COMFYUI_DIR)"
    if not os.access(COMFYUI_PYTHON, os.X_OK):
        return f"no ComfyUI Python at {COMFYUI_PYTHON} (set COMFYUI_PYTHON)"
    return ""


def install_extra_paths() -> str:
    path = os.path.join(COMFYUI_DIR, "extra_model_paths.yaml")
    return path if os.path.isfile(path) else ""


def install_changes(marker: str) -> list[str]:
    """Paths in the ComfyUI install newer than `marker`, outside .venv/ and __pycache__/."""
    out = subprocess.run(["find", COMFYUI_DIR, "-newer", marker, "-not", "-path", "*/.venv/*"],
                         capture_output=True, text=True).stdout.split("\n")
    return [p for p in out if p and "__pycache__" not in p]


def make_base(base: str) -> str:
    """A scratch base directory with this pack in custom_nodes/."""
    os.makedirs(os.path.join(base, "custom_nodes"), exist_ok=True)
    link = os.path.join(base, "custom_nodes", PACK_NAME)
    if not os.path.islink(link):
        os.symlink(ROOT, link)
    return base


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ComfyUI:
    def __init__(self, base: str, env: dict, extra_args=(), log_path=None):
        self.base = base
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = log_path or os.path.join(base, "comfyui.log")
        env = dict(env, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        argv = [COMFYUI_PYTHON, os.path.join(COMFYUI_DIR, "main.py"), "--base-directory", base, "--cpu",
                "--listen", "127.0.0.1", "--port", str(self.port), "--disable-auto-launch",
                "--database-url", f"sqlite:///{os.path.join(base, 'comfyui.db')}", *extra_args]
        self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(argv, cwd=base, env=env, stdout=self._log, stderr=subprocess.STDOUT)

    def log(self) -> str:
        self._log.flush()
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def wait_ready(self, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with {self.proc.returncode}:\n{self.log()[-4000:]}")
            try:
                self.get("/system_stats")
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.5)
        raise TimeoutError(f"ComfyUI did not start in {timeout} s:\n{self.log()[-4000:]}")

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=30) as r:
            return json.loads(r.read() or b"null")

    def post(self, path, body=None):
        data = json.dumps(body if body is not None else {}).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"POST {path}: {e.code} {e.read().decode(errors='replace')}") from None
        return json.loads(raw) if raw else None

    def queue(self, prompt, extra_data=None) -> str:
        body = {"prompt": prompt, "client_id": "mossimage-cpp-tests"}
        if extra_data:
            body["extra_data"] = extra_data
        return self.post("/prompt", body)["prompt_id"]

    def wait_history(self, prompt_id, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            h = self.get(f"/history/{prompt_id}")
            if prompt_id in h and h[prompt_id].get("status", {}).get("completed") is not None:
                entry = h[prompt_id]
                if entry["status"].get("status_str") in ("success", "error"):
                    return entry
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited:\n{self.log()[-4000:]}")
            time.sleep(0.25)
        raise TimeoutError(f"prompt {prompt_id} did not finish in {timeout} s:\n{self.log()[-4000:]}")

    def wait_running(self, prompt_id, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            running = self.get("/queue").get("queue_running", [])
            if any(item[1] == prompt_id for item in running):
                return
            time.sleep(0.05)
        raise TimeoutError(f"prompt {prompt_id} never ran")

    def stop(self, timeout=60) -> int:
        """Ctrl-C, as a user would: the pack's atexit handler shuts moss-server down."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()
        return self.proc.returncode


WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def ui_to_api(workflow: dict, object_info: dict) -> dict:
    """A saved (UI) workflow to the API prompt format, by the nodes' widget order.

    Fails on leftover values and on a required input without one, so a workflow
    out of step with the node schemas is caught. An optional widget without a
    value is left out: that is how a workflow saved before it existed loads.
    """
    links = {link[0]: link for link in workflow["links"]}
    prompt = {}
    for node in workflow["nodes"]:
        if node.get("mode", 0) in (2, 4):  # muted, bypassed
            continue
        info = object_info[node["type"]]
        specs = {**info["input"].get("required", {}), **info["input"].get("optional", {})}
        order = info["input_order"].get("required", []) + info["input_order"].get("optional", [])
        values = list(node.get("widgets_values") or [])
        linked = {i["name"]: i.get("link") for i in node.get("inputs", [])}
        inputs = {}
        for name in order:
            spec = specs[name]
            if isinstance(spec[0], list) or spec[0] in WIDGET_TYPES:
                if not values:
                    if name in info["input"].get("optional", {}):
                        continue
                    raise ValueError(f"node {node['id']} ({node['type']}): no widget value for {name}")
                inputs[name] = values.pop(0)
                if len(spec) > 1 and spec[1].get("control_after_generate"):
                    values.pop(0)
            if linked.get(name) is not None:
                link = links[linked[name]]
                inputs[name] = [str(link[1]), link[2]]
        if values:
            raise ValueError(f"node {node['id']} ({node['type']}): leftover widget values {values}")
        prompt[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    return prompt


def load_workflow_prompt(comfy: ComfyUI) -> dict:
    with open(WORKFLOW, encoding="utf-8") as f:
        return ui_to_api(json.load(f), comfy.get("/object_info"))


def find(prompt: dict, class_type: str) -> dict:
    return next(n for n in prompt.values() if n["class_type"] == class_type)["inputs"]
