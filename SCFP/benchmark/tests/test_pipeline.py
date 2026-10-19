import json
import os

import pytest
import yaml

from scfp_bench import stages
from scfp_bench.answers import extract, is_correct
from scfp_bench.llm import CallLog, Client, ModelSpec

HERE = os.path.dirname(__file__)


# ---------------------------------------------------------------- answers
@pytest.mark.parametrize("text,gold,ok", [
    ("She earns $18 a day.\nFinal answer: 18", "18", True),
    ("Final answer: $1,250.00", "1250", True),
    ("Final answer: 3.5", "3.5", True),
    ("so 7 apples.\nFinal answer: 7 apples", "6", False),
    ("Final answer: 12\n...\nFinal answer: 13", "13", True),   # last final line wins
])
def test_number(text, gold, ok):
    assert is_correct(extract(text, "number").value, gold, "number") is ok


def test_number_fallback_is_flagged():
    e = extract("The total is 42.", "number")
    assert e.value == "42" and e.from_final_line is False


@pytest.mark.parametrize("text,gold", [("Final answer: Yes", "yes"), ("Final answer: no.", "no"),
                                       ("Final answer: True", "yes")])
def test_bool(text, gold):
    assert extract(text, "bool").value == gold


@pytest.mark.parametrize("text,gold", [("Final answer: C", "C"), ("Final answer: (b)", "B"),
                                       ("Final answer: Option D.", "D")])
def test_choice(text, gold):
    assert extract(text, "choice").value == gold


def test_unparseable_is_wrong():
    assert is_correct(extract("I am not sure.", "number").value, "5", "number") is False


# ---------------------------------------------------------------- call log
def test_call_log_caches_and_is_resumable(tmp_path):
    spec = ModelSpec(name="m", provider="mock", model="x")
    msgs = [{"role": "user", "content": "1+1?\n\nEnd your response with a line of the form 'Final answer: <number>'."}]
    log_path = str(tmp_path / "calls.jsonl")
    a = Client(CallLog(log_path)).complete(spec, msgs, 0.0)
    b = Client(CallLog(log_path)).complete(spec, msgs, 0.0)       # fresh process, same log
    assert a["call_id"] == b["call_id"] and a["text"] == b["text"]
    assert sum(1 for _ in open(log_path)) == 1                     # second call came from the log


def test_missing_key_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv("SCFP_TEST_KEY", raising=False)
    spec = ModelSpec(name="m", provider="openai", model="x", api_key_env="SCFP_TEST_KEY")
    with pytest.raises(RuntimeError, match="SCFP_TEST_KEY"):
        Client(CallLog(str(tmp_path / "c.jsonl"))).complete(spec, [{"role": "user", "content": "hi"}], 0.0)


# ---------------------------------------------------------------- agreement
def test_fleiss_kappa_known_values():
    assert stages.fleiss_kappa([[3, 0], [0, 3], [3, 0], [0, 3]]) == pytest.approx(1.0)
    # Standard worked example (10 items, 14 raters, 5 categories): kappa = 0.210
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
             [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
             [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert stages.fleiss_kappa(table) == pytest.approx(0.210, abs=5e-4)
    assert stages.fleiss_kappa([[3, 0], [3, 0]]) is None


# ---------------------------------------------------------------- end to end (mock)
@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    cfg = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "smoke_mock.yaml")))
    cfg["run_dir"] = str(tmp_path_factory.mktemp("run"))
    for s in ("holdout", "filter", "generate", "label", "split"):
        getattr(stages, f"stage_{s}")(cfg)
    return cfg


def test_split_is_by_problem_and_excludes_holdout(mock_run):
    held = {p["problem_id"] for p in stages.read_jsonl(stages.path(mock_run, "holdout.jsonl"))}
    owner = {}
    for name in ("train", "val", "test"):
        for r in stages.read_jsonl(stages.path(mock_run, "splits", f"{name}.jsonl")):
            assert owner.setdefault(r["problem_id"], name) == name
            assert r["problem_id"] not in held


def test_every_trace_points_to_logged_calls(mock_run):
    log = CallLog(stages.path(mock_run, "calls.jsonl"))
    for r in stages.read_jsonl(stages.path(mock_run, "labeled.jsonl")):
        assert set(r["call_ids"]) == {"R0", "F", "Rf"}
        assert all(log.get(c) is not None for c in r["call_ids"].values())


def test_validate_refuses_mock_data(mock_run):
    with pytest.raises(SystemExit):
        stages.stage_validate(mock_run)
    stages.stage_validate(mock_run, allow_mock=True)


def test_annotation_roundtrip(mock_run):
    stages.stage_export(mock_run)
    assign = stages.read_jsonl(stages.path(mock_run, "annotation", "assignments.jsonl"))
    anns = [{"trace_id": a["trace_id"], "annotator": w, "mode": stages.FAILURE_MODES[0]}
            for a in assign for w in a["annotators"]]
    p = stages.path(mock_run, "annotation", "fake_annotations.jsonl")
    stages.write_jsonl(p, anns)
    stages.stage_agree(mock_run, p)
    report = json.load(open(stages.path(mock_run, "annotation", "agreement.json")))
    assert report["needs_adjudication"] == 0
    assert report["final_labels"] == len(assign)


# ---------------------------------------------------------------- exclusion and top-up
def test_require_final_line_excludes_and_counts(mock_run):
    cfg = dict(mock_run, label={"failure_definition": "final_incorrect", "require_final_line": True})
    traces = stages.read_jsonl(stages.path(cfg, "traces.jsonl"))
    traces[0]["final_response"] = "I think it is 7."          # no "Final answer:" line
    stages.write_jsonl(stages.path(cfg, "traces.jsonl"), traces)
    stages.stage_label(cfg)
    summary = json.load(open(stages.path(cfg, "label_summary.json")))
    kept = {r["trace_id"] for r in stages.read_jsonl(stages.path(cfg, "labeled.jsonl"))}
    excl = stages.read_jsonl(stages.path(cfg, "excluded.jsonl"))
    assert traces[0]["trace_id"] not in kept
    assert summary["excluded_no_final_answer_line"] == len(excl) >= 1
    assert summary["traces_generated"] == len(traces)
    assert excl[0]["missing_in"] == ["Rf"] or "Rf" in excl[0]["missing_in"]


def test_topup_keeps_previous_picks(mock_run):
    before = {d["problem_id"] for d in stages.read_jsonl(stages.path(mock_run, "selected.jsonl"))}
    cfg = json.loads(json.dumps(mock_run))
    cfg["filter"]["candidates_per_source"] = {k: v + 10 for k, v in cfg["filter"]["candidates_per_source"].items()}
    stages.stage_filter(cfg)
    after = {d["problem_id"] for d in stages.read_jsonl(stages.path(cfg, "selected.jsonl"))}
    assert before <= after
