"""/perf panel command: timing records, summary (min/avg/max), and raw dump.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_perf_command.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.panel import AiPanel  # noqa: E402
from ludvart.ludvart import Ludvart  # noqa: E402


def _make_ludvart():
    r = Ludvart(["true"])
    r._panel = AiPanel(cols=80, height=10, provider="openai")
    r._render_split = lambda: None
    return r


def _systems(r):
    return [t for kind, t in r._panel.messages if kind == "system"]


def test_perf_timer_records_samples():
    r = _make_ludvart()
    with r._perf_timer("llm_request"):
        pass
    with r._perf_timer("llm_request"):
        pass
    with r._perf_timer("tool:inject_input"):
        pass
    assert len(r._perf["llm_request"]) == 2
    assert len(r._perf["tool:inject_input"]) == 1
    assert all(d >= 0.0 for d in r._perf["llm_request"])
    print("timer records per-op samples: OK")


def test_perf_records_even_on_exception():
    r = _make_ludvart()
    try:
        with r._perf_timer("llm_request"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert len(r._perf["llm_request"]) == 1
    print("timer records even when the block raises: OK")


def test_perf_summary_empty():
    r = _make_ludvart()
    r._handle_slash_command("/perf summary")
    assert any("No performance records" in t for t in _systems(r))
    print("/perf summary (empty): OK")


def test_perf_summary_reports_min_avg_max():
    r = _make_ludvart()
    r._perf["llm_request"] = [1.0, 2.0, 3.0]
    r._perf["tool:inject_input"] = [0.5]
    r._handle_slash_command("/perf summary")
    joined = "\n".join(_systems(r))
    assert "llm_request" in joined
    assert "tool:inject_input" in joined
    # min/avg/max of [1,2,3] == 1.000 / 2.000 / 3.000
    assert "1.000" in joined and "2.000" in joined and "3.000" in joined
    print("/perf summary reports min/avg/max: OK")


def test_perf_dump_lists_raw_records():
    r = _make_ludvart()
    r._perf["llm_request"] = [1.25, 0.5]
    r._handle_slash_command("/perf dump")
    joined = "\n".join(_systems(r))
    assert "llm_request" in joined
    assert "1.250" in joined and "0.500" in joined
    print("/perf dump lists raw records: OK")


def test_perf_default_is_summary():
    r = _make_ludvart()
    r._perf["llm_request"] = [1.0]
    r._handle_slash_command("/perf")
    joined = "\n".join(_systems(r))
    assert "Performance summary" in joined
    print("/perf defaults to summary: OK")


def test_perf_unknown_subcommand():
    r = _make_ludvart()
    r._handle_slash_command("/perf bogus")
    assert any("Usage: /perf" in t for t in _systems(r))
    print("/perf unknown subcommand shows usage: OK")


def main():
    test_perf_timer_records_samples()
    test_perf_records_even_on_exception()
    test_perf_summary_empty()
    test_perf_summary_reports_min_avg_max()
    test_perf_dump_lists_raw_records()
    test_perf_default_is_summary()
    test_perf_unknown_subcommand()
    print("\nALL /perf command tests passed.")


if __name__ == "__main__":
    main()
