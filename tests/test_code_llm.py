"""The Ollama request and the answer checks, tested without a running server."""

import pandas as pd
import pytest
import requests

from src import code_llm
from src.code_llm import answer_schema, build_payload, code_with_retries, parse_answer, think_setting
from src.codebook import load_codebook

FIELDS = load_codebook()
GOOD = ('{"role": "starter", "velo_or_power": "plus", "command_or_approach": "neutral", '
        '"makeup": "not_mentioned", "health_flag": "no", "projection": "some", "overall_tone": "positive"}')


def test_payload_pins_the_settings_that_make_a_run_repeatable():
    schema = answer_schema(FIELDS)
    payload = build_payload("qwen3:8b", "system text", "RHP, 6-2, 190.", schema, think=False)
    assert payload["format"] == schema
    assert payload["options"]["temperature"] == 0
    assert payload["options"]["seed"] == code_llm.SEED
    assert payload["think"] is False
    assert payload["stream"] is False
    assert "RHP, 6-2, 190." in payload["messages"][1]["content"]
    assert "think" not in build_payload("llama3.1:8b", "s", "t", schema, think=None)


def test_schema_limits_every_field_to_its_codes():
    schema = answer_schema(FIELDS)
    assert schema["required"] == [f.name for f in FIELDS]
    assert schema["properties"]["health_flag"]["enum"] == ["yes", "no"]


def test_parse_answer_accepts_good_and_rejects_bad_answers():
    assert parse_answer(GOOD, FIELDS)["velo_or_power"] == "plus"
    with pytest.raises(ValueError):
        parse_answer(GOOD.replace('"plus"', '"elite"'), FIELDS)
    with pytest.raises(ValueError):
        parse_answer('{"role": "starter"}', FIELDS)
    with pytest.raises(ValueError):
        parse_answer("not json", FIELDS)


def test_think_is_only_sent_to_models_that_think():
    assert think_setting(["completion", "thinking"], "qwen3:8b")[0] is False
    assert think_setting(["completion"], "llama3.1:8b")[0] is None
    assert think_setting(None, "qwen3:8b")[0] is False


class FlakyCoder:
    """Fails with a connection error, then returns an invalid answer, then a good one."""

    def __init__(self):
        self.calls = 0

    def answer(self, note):
        self.calls += 1
        if self.calls == 1:
            raise requests.ConnectionError("server busy")
        if self.calls == 2:
            return '{"role": "closer"}', {}
        return GOOD, {"prompt_eval_count": 1000, "eval_count": 60}


def test_retries_recover_from_connection_errors_and_invalid_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(code_llm, "FIRST_WAIT_SECONDS", 0)
    note = pd.Series({"note_id": "N0001", "text": "x"})
    codes, info = code_with_retries(FlakyCoder(), note, FIELDS, tmp_path / "raw.jsonl")
    assert codes is not None and info["attempts"] == 3
    assert len((tmp_path / "raw.jsonl").read_text(encoding="utf-8").splitlines()) == 3  # every attempt logged


def test_a_note_that_never_gets_a_valid_answer_is_reported_not_filled_in(tmp_path, monkeypatch):
    monkeypatch.setattr(code_llm, "FIRST_WAIT_SECONDS", 0)

    class AlwaysBad:
        def answer(self, note):
            return "{}", {}

    note = pd.Series({"note_id": "N0002", "text": "x"})
    codes, info = code_with_retries(AlwaysBad(), note, FIELDS, tmp_path / "raw.jsonl")
    assert codes is None
    assert "invalid answer" in info["error"]
