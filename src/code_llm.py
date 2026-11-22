"""Code every note with a local language model served by Ollama.

One call per note returns all seven fields as JSON. The design choices
are there to make a run trustworthy and repeatable:

  - Ollama's `format` option takes a JSON schema, and the model's output
    is constrained to it, so every answer has the seven fields and only
    codes from the codebook.
  - temperature 0 and a fixed seed, so a rerun gives the same codes.
  - qwen3 can "think" before answering. Thinking is turned off (think =
    false): it is slower, and thinking on or off is a different coder.
  - Connection errors are retried with growing waits. Answers that fail the
    schema check are logged and retried. Notes that fail every attempt
    are listed in the manifest, never filled in silently.
  - Every finished note is appended to a cache file at once, so a run
    that stops halfway resumes where it stopped. Cache entries carry a
    config id (model digest, prompt hash, options), so a changed prompt
    or a re-pulled model never reuses old answers.
  - A manifest records the model digest, Ollama version, options, prompt
    hash and timings, which is what someone needs to reproduce the run.

--mock replaces the model with "truth plus noise" so the pipeline can be
tested without Ollama. Its output is labelled MOCK and can only be
written to outputs_mock/.

Run:  python -m src.code_llm --model qwen3:8b --limit 10    (a quick speed check)
Run on its own, this module writes to outputs_speedcheck/ (or outputs_mock/
with --mock), never to outputs/. Answers go to the shared cache, so the full
run through run_all.py reuses them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.codebook import (
    REPO_ROOT,
    SEED,
    STREAM_MOCK,
    Field,
    load_codebook,
    read_notes,
    rng_for,
    sheet_columns,
)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
OPTIONS = {"temperature": 0, "seed": SEED, "num_ctx": 4096}
MAX_ATTEMPTS = 4
FIRST_WAIT_SECONDS = 2  # waits 2, 4, 8 seconds between attempts
TIMEOUT = (10, 300)     # seconds to connect, seconds to answer (the first call loads the model)
MAX_FAILED_IN_A_ROW = 5  # stop if this many notes in a row fail: the server is likely down
MOCK_LABEL = "MOCK"
MOCK_DIR_NAME = "outputs_mock"
SPEEDCHECK_DIR_NAME = "outputs_speedcheck"


class TooManyFailures(RuntimeError):
    """Raised when several notes in a row fail, so a broken server does not burn hours."""


# ---------------------------------------------------------------- prompt

def build_system_prompt(fields: list[Field]) -> str:
    """The instructions the model sees, written out from codebook.yaml."""
    lines = [
        "You code short amateur baseball scouting notes with a fixed codebook.",
        "Read the note and give exactly one code for each of the seven fields below.",
        "Code only what the note says. Use the rules to settle hard cases.",
        "",
    ]
    for f in fields:
        lines.append(f"Field: {f.name}")
        lines.append(f"Allowed codes: {', '.join(f.values)}")
        lines.append(f"Definition: {f.definition}")
        lines.append("Rules:")
        lines.extend(f"- {rule}" for rule in f.rules)
        lines.append("")
    lines.append("Return one JSON object with all seven fields and nothing else.")
    return "\n".join(lines)


def answer_schema(fields: list[Field]) -> dict:
    """JSON schema for one answer: every field required, each limited to its codes."""
    return {
        "type": "object",
        "properties": {f.name: {"type": "string", "enum": f.values} for f in fields},
        "required": [f.name for f in fields],
        "additionalProperties": False,
    }


def prompt_hash(system_prompt: str, schema: dict) -> str:
    """Short fingerprint of everything the model is told, recorded in the manifest."""
    blob = system_prompt + "\n" + json.dumps(schema, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def user_message(note_text: str) -> str:
    return f"Scouting note:\n{note_text}"


def parse_answer(content: str, fields: list[Field]) -> dict[str, str]:
    """Check one raw answer against the codebook. Raises ValueError if it fails."""
    try:
        answer = json.loads(content)
    except (json.JSONDecodeError, TypeError) as err:
        raise ValueError(f"not valid JSON: {err}") from err
    if not isinstance(answer, dict):
        raise ValueError("answer is not a JSON object")
    expected = {f.name for f in fields}
    if set(answer) != expected:
        raise ValueError(f"wrong keys: missing {sorted(expected - set(answer))}, extra {sorted(set(answer) - expected)}")
    for f in fields:
        if answer[f.name] not in f.values:
            raise ValueError(f"{f.name}={answer[f.name]!r} is not an allowed code")
    return {f.name: answer[f.name] for f in fields}


# ---------------------------------------------------------------- Ollama

class OllamaClient:
    """The few Ollama HTTP endpoints this pipeline needs."""

    def __init__(self, base_url: str = OLLAMA_URL) -> None:
        self.base_url = base_url
        self.session = requests.Session()

    def version(self) -> str:
        r = self.session.get(f"{self.base_url}/api/version", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("version", "unknown")

    def digest(self, model: str) -> str:
        """The digest of the pulled model, which pins the exact weights used."""
        r = self.session.get(f"{self.base_url}/api/tags", timeout=TIMEOUT)
        r.raise_for_status()
        names = {model, f"{model}:latest"}
        for m in r.json().get("models", []):
            if m.get("name") in names or m.get("model") in names:
                return m.get("digest", "unknown")
        raise LookupError(f"Model {model} is not pulled. Run: ollama pull {model}")

    def capabilities(self, model: str) -> list[str] | None:
        """What the model supports (for example 'thinking'), or None on older servers."""
        r = self.session.post(f"{self.base_url}/api/show", json={"model": model}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("capabilities")

    def chat(self, payload: dict) -> dict:
        r = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()


def think_setting(capabilities: list[str] | None, model: str) -> tuple[bool | None, str]:
    """Decide whether to send think=false. Returns (value to send or None, note for the manifest).

    Sending `think` to a model without a thinking mode is an error on
    current Ollama, so it is sent only to models that report thinking.
    """
    if capabilities is None:
        if model.startswith("qwen3"):
            return False, "think=false sent (server did not report capabilities; qwen3 assumed to think)"
        return None, "think not sent (server did not report capabilities)"
    if "thinking" in capabilities:
        return False, "think=false sent (model supports thinking; turned off)"
    return None, "think not sent (model has no thinking mode)"


def build_payload(model: str, system_prompt: str, note_text: str, schema: dict, think: bool | None) -> dict:
    """The /api/chat request for one note. Only the note text is sent, never its true codes."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message(note_text)},
        ],
        "stream": False,
        "format": schema,
        "options": dict(OPTIONS),
    }
    if think is not None:
        payload["think"] = think
    return payload


# ---------------------------------------------------------------- coders

class LlmCoder:
    """Codes a note by calling the model. Knows nothing about true codes."""

    def __init__(self, model: str, client: OllamaClient, fields: list[Field]) -> None:
        self.model = model
        self.label = model
        self.client = client
        self.system_prompt = build_system_prompt(fields)
        self.schema = answer_schema(fields)
        self.digest = client.digest(model)
        self.ollama_version = client.version()
        self.think, self.think_note = think_setting(client.capabilities(model), model)

    def answer(self, note: pd.Series) -> tuple[str, dict]:
        payload = build_payload(self.model, self.system_prompt, note["text"], self.schema, self.think)
        resp = self.client.chat(payload)
        meta = {"prompt_eval_count": resp.get("prompt_eval_count"), "eval_count": resp.get("eval_count")}
        return resp.get("message", {}).get("content", ""), meta


class MockCoder:
    """Truth plus noise, for testing the pipeline without Ollama. NOT a model.

    It reads the true codes, which a real coder never sees, and changes
    each one to a random other code 8% of the time (20% on ambiguous notes).
    """

    SLIP = 0.08
    SLIP_AMBIGUOUS = 0.20

    def __init__(self, fields: list[Field]) -> None:
        self.model = MOCK_LABEL
        self.label = MOCK_LABEL
        self.fields = fields
        self.system_prompt = build_system_prompt(fields)
        self.schema = answer_schema(fields)
        self.digest = "none (MOCK coder, not a model)"
        self.ollama_version = "none (MOCK coder, not a model)"
        self.think, self.think_note = None, "not applicable (MOCK coder)"

    def answer(self, note: pd.Series) -> tuple[str, dict]:
        rng = rng_for(STREAM_MOCK, int(note["note_id"].lstrip("N")))
        slip = self.SLIP_AMBIGUOUS if note["ambiguous"] else self.SLIP
        codes = {}
        for f in self.fields:
            true = note[f"true_{f.name}"]
            if rng.random() < slip:
                others = [v for v in f.values if v != true]
                true = others[int(rng.integers(len(others)))]
            codes[f.name] = true
        return json.dumps(codes), {"prompt_eval_count": None, "eval_count": None}


# ---------------------------------------------------------------- running

def safe_name(label: str) -> str:
    """A file-name-safe version of a model name: qwen3:8b -> qwen3_8b."""
    return label.replace(":", "_").replace("/", "_")


def config_id(coder: LlmCoder | MockCoder) -> str:
    """Fingerprint of everything that could change an answer. Cache entries must match it."""
    blob = json.dumps(
        {
            "model": coder.model,
            "digest": coder.digest,
            "prompt_hash": prompt_hash(coder.system_prompt, coder.schema),
            "options": OPTIONS,
            "think": coder.think,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_cache(path: Path, current_config: str) -> dict[str, dict]:
    """Finished notes from earlier runs with the same config, keyed by note id."""
    done: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record.get("config_id") == current_config:
                        done[record["note_id"]] = record
    return done


def code_with_retries(
    coder: LlmCoder | MockCoder, note: pd.Series, fields: list[Field], raw_log: Path) -> tuple[dict | None, dict]:
    """Code one note, retrying connection errors and invalid answers.

    Returns (codes, info). codes is None if every attempt failed. A 4xx
    error from Ollama (for example a model that is not pulled) is a setup
    problem, not a passing fault, so it stops the run at once.
    """
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            content, meta = coder.answer(note)
        except requests.HTTPError as err:
            if err.response is not None and 400 <= err.response.status_code < 500:
                raise
            last_error = f"HTTP error: {err}"
            append_jsonl(raw_log, {"note_id": note["note_id"], "attempt": attempt, "ok": False, "error": last_error})
        except requests.RequestException as err:
            last_error = f"connection error: {type(err).__name__}: {err}"
            append_jsonl(raw_log, {"note_id": note["note_id"], "attempt": attempt, "ok": False, "error": last_error})
        else:
            seconds = time.perf_counter() - start
            try:
                codes = parse_answer(content, fields)
            except ValueError as err:
                last_error = f"invalid answer: {err}"
                append_jsonl(raw_log, {"note_id": note["note_id"], "attempt": attempt, "ok": False,
                                       "error": last_error, "content": content})
            else:
                append_jsonl(raw_log, {"note_id": note["note_id"], "attempt": attempt, "ok": True,
                                       "content": content})
                return codes, {"attempts": attempt, "seconds": round(seconds, 3), **meta}
        if attempt < MAX_ATTEMPTS:
            time.sleep(FIRST_WAIT_SECONDS * 2 ** (attempt - 1))
    return None, {"attempts": MAX_ATTEMPTS, "error": last_error}


def run(coder: LlmCoder | MockCoder, notes: pd.DataFrame, out_dir: Path, cache_path: Path) -> pd.DataFrame:
    """Code all notes (resuming from the cache), then write the code sheet and manifest."""
    is_mock = isinstance(coder, MockCoder)
    if is_mock and out_dir.name != MOCK_DIR_NAME:
        raise ValueError(f"MOCK output may only be written to {MOCK_DIR_NAME}/, not {out_dir}")
    fields = load_codebook()
    name = safe_name(coder.label)
    raw_log = out_dir / "llm_raw" / f"{name}.jsonl"
    cfg = config_id(coder)
    done = load_cache(cache_path, cfg)
    n_from_cache = sum(nid in done for nid in notes["note_id"])
    started = datetime.now(timezone.utc)
    print(f"[{coder.label}] {len(notes)} notes, {n_from_cache} already in the cache")

    failed: dict[str, str] = {}
    new_seconds: list[float] = []
    failed_in_a_row = 0
    for i, (_, note) in enumerate(notes.iterrows(), start=1):
        if note["note_id"] in done:
            continue
        codes, info = code_with_retries(coder, note, fields, raw_log)
        if codes is None:
            failed[note["note_id"]] = info["error"]
            failed_in_a_row += 1
            if failed_in_a_row >= MAX_FAILED_IN_A_ROW:
                raise TooManyFailures(
                    f"Stopped: {failed_in_a_row} notes in a row failed. Last error: {info['error']}. "
                    "Check that Ollama is running and has enough memory, then run the same command "
                    "again. Finished notes are kept in the cache.")
            continue
        failed_in_a_row = 0
        record = {"note_id": note["note_id"], "config_id": cfg, "codes": codes, **info}
        append_jsonl(cache_path, record)
        done[note["note_id"]] = record
        new_seconds.append(info["seconds"])
        if len(new_seconds) % 50 == 0:
            left = (len(notes) - i) * statistics.median(new_seconds) / 60
            print(f"[{coder.label}] {i}/{len(notes)} notes, about {left:.0f} min left")

    rows = [{"note_id": nid, "coder": coder.label, **done[nid]["codes"]}
            for nid in notes["note_id"] if nid in done]
    sheet = pd.DataFrame(rows, columns=sheet_columns())
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out_dir / f"model_codes_{name}.csv", index=False, lineterminator="\n")

    coded = [done[n] for n in notes["note_id"] if n in done]
    seconds = [r["seconds"] for r in coded]  # includes notes coded in earlier, interrupted runs
    prompt_tokens = [r["prompt_eval_count"] for r in coded if r.get("prompt_eval_count") is not None]
    manifest = {
        "label": MOCK_LABEL if is_mock else "MODEL",
        "note": "MOCK coder: true codes plus random noise, not a language model." if is_mock else "",
        "model": coder.model,
        "model_digest": coder.digest,
        "ollama_version": coder.ollama_version,
        "options": OPTIONS,
        "thinking": coder.think_note,
        "prompt_hash": prompt_hash(coder.system_prompt, coder.schema),
        "config_id": cfg,
        "n_notes": len(notes),
        "n_coded": len(rows),
        "n_from_cache": n_from_cache,
        "n_failed": len(failed),
        "failed": failed,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "new_calls_this_session": len(new_seconds),
        "median_seconds_per_call": round(statistics.median(seconds), 2) if seconds else None,
        "total_model_seconds": round(sum(seconds), 1),
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else None,
        "context_warning": (
            "prompt within 512 tokens of num_ctx: raise num_ctx"
            if prompt_tokens and max(prompt_tokens) > OPTIONS["num_ctx"] - 512 else ""
        ),
    }
    (out_dir / f"manifest_{name}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{coder.label}] coded {len(rows)}/{len(notes)}, failed {len(failed)}")
    return sheet


def make_coder(model: str, mock: bool) -> LlmCoder | MockCoder:
    """Build the mock coder, or connect to Ollama and build the model coder."""
    fields = load_codebook()
    if mock:
        return MockCoder(fields)
    client = OllamaClient()
    try:
        client.version()
    except requests.RequestException:
        sys.exit(f"Ollama is not reachable at {OLLAMA_URL}. Start the Ollama app, then run again.")
    try:
        return LlmCoder(model, client, fields)
    except LookupError as err:
        sys.exit(str(err))


def main() -> None:
    parser = argparse.ArgumentParser(description="Code the notes with a local model through Ollama.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mock", action="store_true", help="truth plus noise, no Ollama needed")
    parser.add_argument("--limit", type=int, default=None, help="code only the first N notes")
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt and its hash, then stop")
    args = parser.parse_args()
    if args.show_prompt:
        fields = load_codebook()
        system_prompt = build_system_prompt(fields)
        print(system_prompt)
        print(f"\nPrompt hash: {prompt_hash(system_prompt, answer_schema(fields))}")
        return
    notes = read_notes()
    if args.limit:
        notes = notes.head(args.limit)
    coder = make_coder(args.model, args.mock)
    # Run on its own, this module never writes to outputs/. Only run_all.py does.
    out_dir = REPO_ROOT / (MOCK_DIR_NAME if args.mock else SPEEDCHECK_DIR_NAME)
    cache_dir = out_dir / "cache" if args.mock else REPO_ROOT / "cache"
    try:
        run(coder, notes, out_dir, cache_dir / f"{safe_name(coder.label)}.jsonl")
    except TooManyFailures as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
