"""Barycentric scatter chart positioning feature extractors across four linguistic dimensions.

Each FE is projected onto 2D by treating the 4 dimensions as compass directions
evenly spaced at 90° (Lexical=E, Syntactic=N, Protocol-level Semantic=W, User-level Semantic=S).
The FE's 2D position is the score-weighted centroid of those unit vectors:

    x = Lexical - Protocol-level Semantic
    y = Syntactic - User-level Semantic

Dimensions (scores 0–10):
  - Lexical:                  surface token features (character/word counts, keyword flags)
  - Syntactic:                grammar/parse-tree structure (depth, terminal nodes, token types)
  - Protocol-level Semantic:  SQL-specific semantics (role of tokens in SQL grammar, semantic tags)
  - User-level Semantic:      meaning derived from user behavior / application-layer context

Run:
    python experiments/fe_taxonomy.py
"""

import argparse
import math
from pathlib import Path

import plotly.graph_objects as go

# fmt: off
FEATURE_EXTRACTORS = {
    # name                    Lexical  Syntactic  Proto. Sem.  User Sem.
    "Li":                    [10,      2,          5,           0],
    "Loginov":               [10,      5,          0,           0],
    "CountVect":             [10,      0,          2,           5],
    "Kakisim":               [0,       5,         10,           0],
    "GAUR (expert)":         [5,       5,          10,          0],
    "GAUR (ChatGPT)":        [5,       5,          10,          0],
}
# fmt: on

DIMENSIONS = ["Lexical", "Syntactic", "Protocol-level Semantic", "User-level Semantic"]

# 4 compass directions: E, N, W, S
_ANGLES_DEG = [0, 90, 180, 270]
_DIRECTIONS = [
    (math.cos(math.radians(a)), math.sin(math.radians(a))) for a in _ANGLES_DEG
]

_PALETTE = {
    "Lexical": "#e07b39",
    "Syntactic": "#5b8dd9",
    "Protocol-level Semantic": "#4caf7d",
    "User-level Semantic": "#9b6dce",
}

_AXIS_LABEL_OFFSET = 2  # how far past the arrow tip to place the label

_LABEL_OFFSET = 24  # pixels from dot centre to label anchor

# Label offset (ax, ay) in pixels relative to the dot. Positive ay = downward.
# fmt: off
_TEXT_OFFSETS: dict[str, tuple[int, int]] = {
    "Li":               ( _LABEL_OFFSET, -_LABEL_OFFSET),
    "Loginov":          ( _LABEL_OFFSET, -_LABEL_OFFSET),
    "CountVect":        ( _LABEL_OFFSET, -_LABEL_OFFSET),
    "Kakisim":          (             0, -_LABEL_OFFSET),
    "GAUR (expert)":    (             0,  _LABEL_OFFSET),
    "GAUR (ChatGPT)":   ( _LABEL_OFFSET, -_LABEL_OFFSET),
}
# fmt: on


def _dominant_category(scores: list[int]) -> str:
    return DIMENSIONS[scores.index(max(scores))]


def _project(scores: list[int]) -> tuple[float, float]:
    """Weighted centroid of the 4 unit-direction vectors."""
    x = sum(s / 2 * dx for s, (dx, _) in zip(scores, _DIRECTIONS))
    y = sum(s / 2 * dy for s, (_, dy) in zip(scores, _DIRECTIONS))
    return x, y


def build_figure() -> go.Figure:
    fig = go.Figure()

    # ---- axis arrows and labels ----
    arrow_len = 3.5
    label_pos = {
        "Lexical": (arrow_len * _AXIS_LABEL_OFFSET * 0.9, 0),
        "Syntactic": (0, arrow_len * 1.3),
        "Protocol-level Semantic": (-arrow_len * _AXIS_LABEL_OFFSET * 0.9, 0),
        "User-level Semantic": (0, -arrow_len * 1.3),
    }
    for dim, (dx, dy) in zip(DIMENSIONS, _DIRECTIONS):
        color = _PALETTE[dim]
        lx, ly = label_pos[dim]
        fig.add_annotation(
            x=dx * arrow_len,
            y=dy * arrow_len,
            ax=0,
            ay=0,
            axref="x",
            ayref="y",
            xref="x",
            yref="y",
            arrowhead=3,
            arrowwidth=1.5,
            arrowcolor=color,
            showarrow=True,
        )
        fig.add_annotation(
            x=lx,
            y=ly,
            text="<b>"
            + dim.replace(
                "Protocol-level Semantic", "Protocol-level<br>Semantic"
            ).replace("User-level Semantic", "User-level<br>Semantic")
            + "</b>",
            showarrow=False,
            font=dict(size=13, color=color),
            xref="x",
            yref="y",
        )

    # ---- one trace per FE ----
    added_cats: set[str] = set()
    for name, scores in FEATURE_EXTRACTORS.items():
        cat = _dominant_category(scores)
        color = _PALETTE[cat]
        px, py = _project(scores)
        show_legend_group = cat not in added_cats
        added_cats.add(cat)

        fig.add_trace(
            go.Scatter(
                x=[px],
                y=[py],
                mode="markers",
                name=cat,
                legendgroup=cat,
                showlegend=show_legend_group,
                marker=dict(color=color, size=11, line=dict(color="white", width=1.5)),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    + "<br>".join(f"{d}: {s}" for d, s in zip(DIMENSIONS, scores))
                    + "<extra></extra>"
                ),
            )
        )

        ax, ay = _TEXT_OFFSETS.get(name, (0, -_LABEL_OFFSET))
        fig.add_annotation(
            x=px,
            y=py,
            ax=ax,
            ay=ay,
            axref="pixel",
            ayref="pixel",
            xref="x",
            yref="y",
            text=name,
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.4,
            arrowcolor="#cccccc",
            standoff=7,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        )

    lim = 14
    fig.update_layout(
        xaxis=dict(
            range=[-lim, lim],
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            scaleanchor="y",
        ),
        yaxis=dict(
            range=[-lim, lim],
            zeroline=False,
            showgrid=False,
            showticklabels=False,
        ),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=800,
        height=800,
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Barycentric FE taxonomy scatter chart"
    )
    parser.add_argument(
        "--output-dir",
        default="output/results/fe_taxonomy",
        help="Output directory for the PDF figure",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    out = output_dir / "fe_taxonomy.pdf"
    fig.write_image(str(out), format="pdf")
    print(f"Saved → {out}")
