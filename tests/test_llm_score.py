from graph_retriever.llm_score import QID, parse_run, repair


def test_parse_run_setting_with_equals():
    assert parse_run("llm:div=2=out/a.jsonl") == ("llm", "div=2", __import__("pathlib").Path("out/a.jsonl"))
    assert parse_run("llm=out/a.jsonl")[:2] == ("llm", "")


def test_query_id_pattern():
    m = QID.match("pisa-pisa_seed0_test-12")
    assert (m["city"], m["seed"], m["split"], m["q"]) == ("pisa", "0", "test", "12")
    assert QID.match("pisa-rome_seed0_test-12") is None


def test_repair_drops_invalid_and_appends_missing():
    id_of = {"a": 0, "b": 1, "c": 2, "d": 3, "z": 9}
    stats = {"invalid": 0, "missing": 0}
    # 'z' is a real item but not a candidate, 'q' unknown, 'b' duplicated
    out = repair(["b", "z", "q", "b", "a"], [0, 1, 2, 3], id_of, stats)
    assert out == [1, 0, 2, 3]
    assert stats == {"invalid": 3, "missing": 2}
