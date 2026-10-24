import os
import sys

import pytest

sys.dont_write_bytecode = True  # nothing lands next to the sources (or in an install the tests import)

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
for p in (ROOT, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from helpers import make_fake_server  # noqa: E402


@pytest.fixture
def fake_server(tmp_path):
    return make_fake_server(str(tmp_path / "bin"))


@pytest.fixture
def fake_env(tmp_path):
    """Environment for the fake: tests add FAKE_MOSS_* entries before starting it."""
    env = dict(os.environ)
    env["FAKE_MOSS_RECORD"] = str(tmp_path / "requests.jsonl")
    env["FAKE_MOSS_STEP_DELAY"] = "0.01"
    return env
