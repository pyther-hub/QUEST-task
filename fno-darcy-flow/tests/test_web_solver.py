import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_web_solver_node_tests():
    r = subprocess.run(["node", "--test", str(WEB / "solver.test.js")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
