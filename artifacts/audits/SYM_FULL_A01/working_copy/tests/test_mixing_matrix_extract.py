import json, os
import pandas as pd
import pytest

from scripts.mixing_matrix_extract import (
    load_mu_from_json,
    locking_ratios,
    write_outputs,
)

def test_load_and_ratios(tmp_path):
    j = tmp_path / "ok.json"
    payload = {
        "N":    {"ROD": 0.75, "CM": -0.50},
        "divN": {"ROD": 0.20, "CM": -0.13333333333333333},
    }
    j.write_text(json.dumps(payload))
    mu = load_mu_from_json(str(j))
    rN, rDiv = locking_ratios(mu)
    assert pytest.approx(rN, rel=0, abs=1e-12) == 2/3
    assert pytest.approx(rDiv, rel=0, abs=1e-12) == 2/3

def test_write_outputs_ok(tmp_path):
    mu = {
        "N":    {"ROD": 0.75, "CM": -0.50},
        "divN": {"ROD": 0.20, "CM": -0.13333333333333333},
    }
    out_csv  = tmp_path / "mixing_matrix.csv"
    out_meta = tmp_path / "mixing_matrix_meta.json"
    rc = write_outputs(mu, str(out_csv), str(out_meta), tol=1e-6, require_lock=True, print_summary=False)
    assert rc == 0
    assert out_csv.exists() and os.path.exists(str(out_csv) + ".md5")
    assert out_meta.exists() and os.path.exists(str(out_meta) + ".md5")

    df = pd.read_csv(out_csv)
    assert set(df.columns) == {"entry", "mu_ROD", "mu_CM"}
    assert set(df["entry"]) == {"N", "divN"}

    meta = json.loads(out_meta.read_text())
    assert meta["locking_check"]["lock_ok"] is True
    assert pytest.approx(meta["locked_weights"]["w_ROD_star_over_w_CM_star"], abs=1e-12) == 2/3

def test_write_outputs_fail_when_mismatch(tmp_path):
    mu_bad = {
        "N":    {"ROD": 0.75, "CM": -0.50},
        "divN": {"ROD": 0.20, "CM":  0.35},  # mismatch on purpose
    }
    out_csv  = tmp_path / "mixing_matrix.csv"
    out_meta = tmp_path / "mixing_matrix_meta.json"
    rc = write_outputs(mu_bad, str(out_csv), str(out_meta), tol=1e-9, require_lock=True, print_summary=False)
    assert rc == 1  # should fail in strict mode
