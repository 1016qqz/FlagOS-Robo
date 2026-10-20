"""Tests for the per-step plain-text log emitter.

Why this is its own module: the JSONL dump is consumed by
``tools/compare_precision.py``, which parses JSON. This log is consumed by a
``grep -E "^\\[step (199|200)\\]"`` over the file -- a *textual* expectation,
and one a logging-framework timestamp prefix silently breaks. So the tests
here assert on the exact rendered bytes and drive that real regex, rather than
re-checking record fields the JSONL tests already cover.
"""

from __future__ import annotations

import re

import pytest

from flagos_robo.utils import precision_dump as pd

#: The regex from the acceptance step, verbatim. A module constant so any
#: change to it is a visible diff rather than a quiet weakening inside one test.
ACCEPTANCE_RE = re.compile(r"^\[step (199|200)\]")


def test_line_starts_at_column_zero_with_the_step_marker():
    line = pd.format_step_log_line(199, loss=0.5)

    assert line.startswith("[step 199]")
    assert ACCEPTANCE_RE.match(line)


def test_the_acceptance_grep_matches_the_last_two_steps_of_a_real_file(tmp_path):
    """Drives the actual acceptance regex over a file written step by step."""
    for step in range(197, 201):
        pd.append_step_log(tmp_path, step, loss=0.5, extra={"grad_norm": 1.0})

    lines = (tmp_path / "rank0.log").read_text(encoding="utf-8").splitlines()
    hits = [line for line in lines if ACCEPTANCE_RE.match(line)]

    assert len(lines) == 4
    assert len(hits) == 2, lines
    assert hits[0].startswith("[step 199]")
    assert hits[1].startswith("[step 200]")


def test_a_logger_style_timestamp_prefix_would_not_match():
    """Pins *why* this is a bare file append and not a ``logger`` call."""
    bare = pd.format_step_log_line(200, loss=0.5)
    prefixed = f"[2026-10-20 15:54:37,238 FlagOS-Robo train.py:649 INFO] {bare}"

    assert ACCEPTANCE_RE.match(bare)
    assert not ACCEPTANCE_RE.match(prefixed)


def test_full_float_precision_is_preserved():
    """A log rounded for human eyes cannot be cross-checked against the JSONL."""
    loss = 0.4957215487957001
    line = pd.format_step_log_line(20, loss=loss, extra={"grad_norm": 4.299605846405029})

    assert repr(loss) in line
    assert repr(4.299605846405029) in line


def test_appends_rather_than_truncates(tmp_path):
    pd.append_step_log(tmp_path, 1, loss=1.0)
    pd.append_step_log(tmp_path, 2, loss=2.0)

    text = (tmp_path / "rank0.log").read_text(encoding="utf-8")

    assert text.splitlines() == ["[step 1] loss=1.0", "[step 2] loss=2.0"]


def test_only_rank_zero_writes_by_default(tmp_path):
    """Six ranks appending to one file would interleave into garbage."""
    assert pd.append_step_log(tmp_path, 1, loss=1.0, rank=3) is None
    assert not (tmp_path / "rank0.log").exists()

    assert pd.append_step_log(tmp_path, 1, loss=1.0, rank=0) is not None


def test_every_rank_can_be_logged_when_explicitly_asked(tmp_path):
    assert pd.append_step_log(tmp_path, 1, loss=1.0, rank=3, only_rank=None) is not None


def test_parent_directory_is_created(tmp_path):
    path = pd.append_step_log(tmp_path / "nested" / "run", 1, loss=1.0)

    assert path is not None and path.exists()
    assert path.name == "rank0.log"


def test_extra_cannot_shadow_step_or_loss():
    line = pd.format_step_log_line(7, loss=0.25, extra={"step": 999, "loss": 9.0})

    assert line == "[step 7] loss=0.25"


def test_extra_keys_are_rendered_in_a_stable_order():
    """Unstable key order makes two runs' logs differ textually for no reason."""
    a = pd.format_step_log_line(1, loss=0.5, extra={"lr": 1e-5, "grad_norm": 2.0})
    b = pd.format_step_log_line(1, loss=0.5, extra={"grad_norm": 2.0, "lr": 1e-5})

    assert a == b
    assert a.index("grad_norm") < a.index("lr")


def test_non_numeric_extra_is_rendered_without_float_coercion():
    line = pd.format_step_log_line(1, loss=0.5, extra={"phase": "warmup", "ok": True})

    assert "phase=warmup" in line
    assert "ok=True" in line


def test_loss_may_be_omitted():
    assert pd.format_step_log_line(5) == "[step 5]"


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "", "   "])
def test_gate_is_off_for_falsey_values(value):
    assert pd.step_log_dir_from_env({pd.STEP_LOG_ENV_VAR: value}) is None


def test_gate_is_off_when_unset():
    assert pd.step_log_dir_from_env({}) is None


def test_gate_returns_the_directory_and_strips_whitespace():
    got = pd.step_log_dir_from_env({pd.STEP_LOG_ENV_VAR: "  /tmp/run7  "})

    assert got is not None and str(got) == "/tmp/run7"


def test_the_two_gates_agree_on_what_off_means():
    """If one gate treated "off" as a directory name and the other as disabled,
    a run would produce one artifact and not the other -- and the missing one
    would look like a crash rather than a config mistake."""
    for value in ("0", "false", "no", "off", "", "  "):
        dump = pd.dump_dir_from_env({pd.DUMP_ENV_VAR: value})
        log = pd.step_log_dir_from_env({pd.STEP_LOG_ENV_VAR: value})
        assert (dump is None) == (log is None), value

    dump = pd.dump_dir_from_env({pd.DUMP_ENV_VAR: "/tmp/d"})
    log = pd.step_log_dir_from_env({pd.STEP_LOG_ENV_VAR: "/tmp/d"})
    assert str(dump) == str(log) == "/tmp/d"
