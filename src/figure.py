"""Draw the two-panel summary figure from the scored tables.

Left panel, like for like: for each field, the keyword list's and the
model's mean Cohen's kappa with each of the three human coders, beside the
human coders' mean kappa with each other (the human ceiling). Lines are
bootstrap 95% intervals. The grey tick is the decision rule's threshold:
the ceiling minus .10. A machine coder left of its tick fails that clause.

Right panel: on the test notes where the model and the consensus
disagree, the share where the model matched the true code, where the
consensus did, and where neither did, by the type of sentence the field
was written in. The hedged health row is where the simulated coders share
a planted habit, so it is where the consensus is expected to be wrong.

The figure is built only from the CSV tables, so it can be redrawn
without rerunning anything:  python -m src.figure --out outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.codebook import REPO_ROOT, SENTENCE_TYPES, field_names

LABELS = {
    "role": "Role",
    "velo_or_power": "Velocity or power",
    "command_or_approach": "Command or approach",
    "makeup": "Makeup",
    "health_flag": "Health flag",
    "projection": "Projection",
    "overall_tone": "Overall tone",
}
# Colors follow the entity across both panels: blue = model, aqua = human
# coders, orange = keyword list. Extra models take the next palette slots.
MODEL_COLORS = ["#2a78d6", "#eda100", "#e87ba4"]
HUMAN_COLOR = "#1baf7a"
KEYWORD_COLOR = "#eb6834"
NEITHER_COLOR = "#c3c2b7"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def kappa_panel(ax: Axes, agreement: pd.DataFrame, models: list[str]) -> None:
    """Left panel: mean kappa with each human coder, by field, beside the human ceiling."""
    fields = field_names()
    table = agreement[agreement["notes"] == "all test notes"]
    each = table[table["reference"] == "each human coder (mean)"]
    ceiling = table[table["coder"] == "human_ceiling"].set_index("field").loc[fields]
    series = [(each[each["coder"] == m], MODEL_COLORS[i], "o", f"{m}: mean kappa with each human coder")
              for i, m in enumerate(models)]
    series += [(each[each["coder"] == "keyword"], KEYWORD_COLOR, "s", "Keyword list: mean kappa with each human coder"),
               (ceiling.reset_index(), HUMAN_COLOR, "D", "Human coders: mean kappa with each other")]
    step = 0.22
    offsets = [step * (len(series) - 1) / 2 - step * i for i in range(len(series))]
    for (rows, color, marker, label), dy in zip(series, offsets):
        rows = rows.set_index("field").loc[fields]
        y = [len(fields) - 1 - i + dy for i in range(len(fields))]
        ax.hlines(y, rows["kappa_lo"], rows["kappa_hi"], color=color, linewidth=2)
        ax.plot(rows["kappa"], y, marker, color=color, markersize=8, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
    for i, f in enumerate(fields):
        y = len(fields) - 1 - i
        ax.vlines(ceiling.loc[f, "kappa"] - 0.10, y - 0.42, y + 0.42, color=INK_2, linewidth=1.5,
                  label="Rule threshold: human ceiling minus .10" if i == 0 else None)
    lows = pd.concat([r for r, *_ in series])["kappa_lo"]
    ax.set_xlim(min(0.4, int(lows.min() * 10) / 10), 1)
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels([LABELS[f] for f in reversed(fields)])
    ax.set_ylim(-0.6, len(fields) - 0.4)
    ax.set_xlabel("Mean Cohen's kappa with each human coder (human coders: with each other), test notes")
    ax.set_title("Agreement with individual human coders, by field", loc="left", fontsize=12, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9)


def audit_panel(ax: Axes, audit: pd.DataFrame, model: str) -> None:
    """Right panel: who matched the true code when the model and the consensus disagree, by sentence type."""
    order = [*SENTENCE_TYPES, "all"]
    labels = {**SENTENCE_TYPES, "all": "All sentences"}
    rows = audit[(audit["coder"] == model) & (audit["field"] == "all_fields")].set_index("sentence_type").loc[order]
    who = "MOCK" if model == "MOCK" else "Model"
    parts = [("share_coder_right", MODEL_COLORS[0], f"{who} right", "#ffffff"),
             ("share_consensus_right", HUMAN_COLOR, "Consensus right", INK),
             ("share_neither", NEITHER_COLOR, "Neither right", INK)]
    y = [len(order) - 1 - i for i in range(len(order))]
    left = pd.Series(0.0, index=rows.index)
    for col, color, label, text_color in parts:
        values = rows[col].fillna(0)
        ax.barh(y, values, left=left, color=color, label=label, height=0.62,
                edgecolor=SURFACE, linewidth=2)
        for yi, v, l0 in zip(y, values, left):
            if v >= 0.12:
                ax.text(l0 + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=8.5, color=text_color)
        left = left + values
    for yi, n in zip(y, rows["n_disagree"]):
        ax.text(1.02, yi, f"n = {int(n)}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[t] for t in order])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of model-consensus disagreements, all fields pooled (test notes)")
    ax.set_title("When model and consensus disagree, who was right?", loc="left", fontsize=12, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9)


def style(ax: Axes) -> None:
    """Quiet axes: no box, light grid, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw(out_dir: Path, models: list[str], mock: bool = False) -> Path:
    """Draw the figure from the tables in out_dir and save it as PNG."""
    agreement = pd.read_csv(out_dir / "agreement_by_field.csv")
    audit = pd.read_csv(out_dir / "disagreement_audit.csv")
    # A bare Figure (not pyplot) draws straight to a file, with no window or GUI backend.
    fig = Figure(figsize=(15, 6.6))
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor(SURFACE)
    kappa_panel(left, agreement, models)
    audit_panel(right, audit, models[0])
    for ax in (left, right):
        style(ax)
    if mock:
        fig.suptitle("MOCK RUN: the 'model' here is true codes plus random noise, not a language model",
                     color="#d03b3b", fontsize=13, fontweight="bold")
        fig.text(0.5, 0.5, "MOCK", fontsize=120, color="#d03b3b", alpha=0.08,
                 ha="center", va="center", rotation=20)
    else:
        fig.suptitle(f"Model: {', '.join(models)}. Synthetic notes: the numbers show the method, not a finding.",
                     color=INK_2, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "figures" / "agreement_by_field.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the summary figure from the scored tables.")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    out_dir = REPO_ROOT / args.out
    decisions = pd.read_csv(out_dir / "decision_by_field.csv")
    models = list(dict.fromkeys(decisions["model"]))
    path = draw(out_dir, models, mock=models == ["MOCK"])
    print(f"Wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
