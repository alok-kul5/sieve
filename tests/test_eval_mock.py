import json
import subprocess
import sys
from pathlib import Path

FIX = Path(__file__).parent / "fixtures" / "mini_test.jsonl"


def test_mock_eval_end_to_end(tmp_path):
    cmd = [sys.executable, "-m", "sieve.evaluate", "--backend", "mock", "--name", "mock", "--test", str(FIX),
           "--train", str(FIX), "--out", str(tmp_path)]
    subprocess.run(cmd, check=True, cwd=Path(__file__).parents[1])
    res = json.loads((tmp_path / "mock.json").read_text())
    assert res["n"] == 8
    assert res["json_valid"] == 7 / 8  # every 7th output is garbage
    assert 0.5 < res["fact_f1"] < 1.0
    preds = (tmp_path / "mock.preds.jsonl").read_text().splitlines()
    assert len(preds) == 8
