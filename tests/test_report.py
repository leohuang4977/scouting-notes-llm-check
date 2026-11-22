"""results.md must say plainly when there is no git commit to check the plan against."""

from src import report


def test_no_repository_is_reported_plainly(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)
    state = report.git_state()
    assert state["commit"] is None
    lines = report.git_lines(state, ["abc123def456"])
    assert "cannot show that the plan was written down before it ran" in lines[0]
