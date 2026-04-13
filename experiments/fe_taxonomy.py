"""Barycentric scatter chart positioning feature extractors across four linguistic dimensions.

Each FE is projected onto 2D by treating the 4 dimensions as compass directions
evenly spaced at 90° (Lexical=E, Syntactic=N, Formal Semantic=W, Distrib. Sem.=S).
The FE's 2D position is the score-weighted centroid of those unit vectors:

    x = Lexical - Formal Semantic
    y = Syntactic - Distributional Semantic

Dimensions (scores 0–10):
  - Lexical:               surface token features (character/word counts, keyword flags)
  - Syntactic:             grammar/parse-tree structure (depth, terminal nodes, token types)
  - Formal Semantic:       SQL-specific semantics (role of tokens in SQL grammar, semantic tags)
  - Distributional Sem.:   meaning from co-occurrence / large pre-trained corpora

Run:
    python experiments/fe_taxonomy.py
"""

import argparse
import math
from pathlib import Path

import plotly.graph_objects as go

# fmt: off
FEATURE_EXTRACTORS = {
    # name                    Lexical  Syntactic  Formal Sem.  Distrib. Sem.
    "Li":                    [10,      2,          7,           0], # Derived from taxonomy_scores.csv
    "Loginov":               [10,      0,          0,           0], # Derived from taxonomy_scores.csv
    "CountVect":             [10,      0,          0,           0], # Pure lexical
    "Kakisim":               [6,       10,         6,           0], # See analysis
    "GAUR (expert)":         [4,       3,          10,          0], # Derived from taxonomy_scores.csv
    "GAUR (ChatGPT)":        [5,       5,          10,          0], # Derived from taxonomy_scores.csv
    "SecureBERT":            [0,       0,          4,           10],# Distributional and some code concept through tokeniser / corpus
    "RoBERTa":               [0,       0,          0,           10],# Pure distributional
    "CodeBERT":              [0,       6,          6,           10],# Distributional with AST capabilities (thus semantic too)
    "CodeT5":                [0,       6,          6,           10],# Distributional with AST capabilities (thus semantic too)
    "Flan-T5":               [0,       0,          0,           10],# Pure distributional  
    "SentBERT":              [0,       0,          0,           10],# Pure distributional
    "LLM2Vec":               [0,       0,          0,           10],# Pure distributional
}
# fmt: on

DIMENSIONS = ["Lexical", "Syntactic", "Formal Semantic", "Distributional Semantic"]

# 4 compass directions: E, N, W, S
_ANGLES_DEG = [0, 90, 180, 270]
_DIRECTIONS = [
    (math.cos(math.radians(a)), math.sin(math.radians(a))) for a in _ANGLES_DEG
]

_PALETTE = {
    "Lexical": "#e07b39",
    "Syntactic": "#5b8dd9",
    "Formal Semantic": "#4caf7d",
    "Distributional Semantic": "#9b6dce",
}

_AXIS_LABEL_OFFSET = 2  # how far past the arrow tip to place the label

# Label position per FE. Options: "top center", "top left", "top right",
# "bottom center", "bottom left", "bottom right", "middle left", "middle right"
# fmt: off
_TEXT_POSITIONS: dict[str, str] = {
    "Li":           "top right",
    "Loginov":      "bottom left",
    "CountVect":    "top center",
    "Kakisim":          "top center",
    "GAUR (expert)":    "top center",
    "GAUR (ChatGPT)":   "bottom center",
    "SecureBERT":   "top center",
    "RoBERTa":      "top right",
    "CodeBERT":     "top left",
    "CodeT5":       "bottom right",
    "Flan-T5":      "bottom left",
    "SentBERT":     "top left",
    "LLM2Vec":      "bottom right",
}
# fmt: on


def _dominant_category(scores: list[int]) -> str:
    return DIMENSIONS[scores.index(max(scores))]


def _project(scores: list[int]) -> tuple[float, float]:
    """Weighted centroid of the 4 unit-direction vectors."""
    x = sum(s * dx for s, (dx, _) in zip(scores, _DIRECTIONS))
    y = sum(s * dy for s, (_, dy) in zip(scores, _DIRECTIONS))
    return x, y


def build_figure() -> go.Figure:
    fig = go.Figure()

    # ---- axis arrows and labels ----
    arrow_len = 3.5
    label_pos = {
        "Lexical": (arrow_len * _AXIS_LABEL_OFFSET, 0),
        "Syntactic": (0, arrow_len * _AXIS_LABEL_OFFSET),
        "Formal Semantic": (-arrow_len * _AXIS_LABEL_OFFSET, 0),
        "Distributional Semantic": (0, -arrow_len * _AXIS_LABEL_OFFSET),
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
            text=f"<b>{dim}</b>",
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
                mode="markers+text",
                name=cat,
                legendgroup=cat,
                showlegend=show_legend_group,
                marker=dict(color=color, size=11, line=dict(color="white", width=1.5)),
                text=[name],
                textposition=_TEXT_POSITIONS.get(name, "top center"),
                textfont=dict(size=10),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    + "<br>".join(f"{d}: {s}" for d, s in zip(DIMENSIONS, scores))
                    + "<extra></extra>"
                ),
            )
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
    parser = argparse.ArgumentParser(description="Barycentric FE taxonomy scatter chart")
    parser.add_argument(
        "--output-dir",
        default="output/experiments/fe_taxonomy",
        help="Output directory for the PDF figure",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    out = output_dir / "fe_taxonomy.pdf"
    fig.write_image(str(out), format="pdf")
    print(f"Saved → {out}")
