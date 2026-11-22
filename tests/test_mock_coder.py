"""The MOCK coder must be labelled MOCK everywhere and kept out of outputs/."""

import json

import pytest

from src import code_llm
from src.code_llm import MOCK_LABEL, MockCoder, TooManyFailures, answer_schema, build_system_prompt, run
from src.codebook import load_codebook, read_notes


@pytest.fixture
def few_notes():
    return read_notes().head(20)


def test_mock_output_is_labelled_mock(tmp_path, few_notes):
    out = tmp_path / "outputs_mock"
    sheet = run(MockCoder(load_codebook()), few_notes, out, out / "cache" / "MOCK.jsonl")
    assert MOCK_LABEL == "MOCK"
    assert set(sheet["coder"]) == {"MOCK"}
    assert (out / "model_codes_MOCK.csv").exists()
    manifest = json.loads((out / "manifest_MOCK.json").read_text(encoding="utf-8"))
    assert manifest["label"] == "MOCK"
    assert "not a language model" in manifest["note"]
    assert "MOCK" in manifest["model_digest"]


def test_mock_output_cannot_go_to_the_real_outputs_folder(tmp_path, few_notes):
    with pytest.raises(ValueError, match="MOCK"):
        run(MockCoder(load_codebook()), few_notes, tmp_path / "outputs", tmp_path / "cache.jsonl")


def test_a_second_run_resumes_from_the_cache_with_the_same_codes(tmp_path, few_notes):
    out = tmp_path / "outputs_mock"
    cache = out / "cache" / "MOCK.jsonl"
    first = run(MockCoder(load_codebook()), few_notes, out, cache)
    second = run(MockCoder(load_codebook()), few_notes, out, cache)
    manifest = json.loads((out / "manifest_MOCK.json").read_text(encoding="utf-8"))
    assert manifest["n_from_cache"] == 20
    assert manifest["new_calls_this_session"] == 0
    assert first.equals(second)


class FakeModel:
    """Stands in for a real model coder: same attributes, canned answers."""

    def __init__(self, answer='{}'):
        fields = load_codebook()
        self.model = self.label = "fake:1b"
        self.system_prompt, self.schema = build_system_prompt(fields), answer_schema(fields)
        self.digest, self.ollama_version = "sha256:fake", "0.0.0"
        self.think, self.think_note = None, "none"
        self._answer = answer

    def answer(self, note):
        return self._answer, {"prompt_eval_count": 10, "eval_count": 5}


def test_the_speed_check_never_writes_to_outputs(tmp_path, monkeypatch):
    good = ('{"role": "starter", "velo_or_power": "plus", "command_or_approach": "neutral", '
            '"makeup": "not_mentioned", "health_flag": "no", "projection": "some", "overall_tone": "positive"}')
    monkeypatch.setattr(code_llm, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(code_llm, "make_coder", lambda model, mock: FakeModel(good))
    monkeypatch.setattr("sys.argv", ["code_llm", "--model", "fake:1b", "--limit", "3"])
    code_llm.main()
    assert (tmp_path / "outputs_speedcheck" / "model_codes_fake_1b.csv").exists()
    assert not (tmp_path / "outputs").exists()


def test_a_run_stops_after_several_notes_in_a_row_fail(tmp_path, monkeypatch, few_notes):
    monkeypatch.setattr(code_llm, "FIRST_WAIT_SECONDS", 0)
    with pytest.raises(TooManyFailures):
        run(FakeModel("{}"), few_notes, tmp_path / "outputs_speedcheck", tmp_path / "cache.jsonl")
