"""IHEval metrics aggregation."""

from src.evaluation.external.iheval.metrics import aggregate_iheval_metrics


def _r(task: str, setting: str, score: float) -> dict:
    return {"task": task, "setting": setting, "score": score}


def test_per_setting_per_task_means() -> None:
    records = [
        _r("single-turn", "aligned", 0.8), _r("single-turn", "aligned", 0.6),
        _r("translation", "aligned", 0.9),
        _r("single-turn", "conflict", 0.4),
        _r("translation", "conflict", 0.5),
        _r("single-turn", "reference", 0.7),
        _r("translation", "reference", 0.85),
    ]
    m = aggregate_iheval_metrics(records, tasks_run=("single-turn", "translation"))
    assert m["by_setting"]["aligned"]["single-turn"] == 0.7   # (0.8 + 0.6) / 2
    assert m["by_setting"]["aligned"]["translation"] == 0.9
    assert m["by_setting"]["conflict"]["single-turn"] == 0.4


def test_per_setting_macro() -> None:
    records = [
        _r("single-turn", "conflict", 0.4),
        _r("translation", "conflict", 0.6),
    ]
    m = aggregate_iheval_metrics(records, tasks_run=("single-turn", "translation"))
    assert m["per_setting_macro"]["conflict"] == 0.5


def test_iheval_score_is_macro_conflict() -> None:
    records = [
        _r("single-turn", "conflict", 0.4),
        _r("translation", "conflict", 0.6),
    ]
    m = aggregate_iheval_metrics(records, tasks_run=("single-turn", "translation"))
    assert m["iheval_score"] == 0.5


def test_ih_following_per_task_and_macro() -> None:
    records = [
        _r("single-turn", "conflict", 0.4),
        _r("single-turn", "reference", 0.8),
        _r("translation", "conflict", 0.6),
        _r("translation", "reference", 0.6),
    ]
    m = aggregate_iheval_metrics(records, tasks_run=("single-turn", "translation"))
    assert m["ih_following"]["single-turn"] == 0.5   # 0.4 / 0.8
    assert m["ih_following"]["translation"] == 1.0   # 0.6 / 0.6
    assert m["ih_following"]["macro"] == 0.75


def test_settings_run_recorded() -> None:
    records = [_r("single-turn", "aligned", 0.5)]
    m = aggregate_iheval_metrics(records, tasks_run=("single-turn",))
    assert m["settings_run"] == ["aligned"]
    assert m["tasks_run"] == ["single-turn"]
    assert m["tool_use_status"].startswith("deferred")
