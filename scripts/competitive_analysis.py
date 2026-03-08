"""
Competitive Analysis Script for ICDI-X Paper
=============================================
Generates a LaTeX comparison table: ICDI-X vs Perplexity vs NotebookLM vs Plain RAG.

Usage:
    python scripts/competitive_analysis.py

Output:
    - prints LaTeX table to stdout (paste into paper Section 5)
    - saves scripts/results/competitive_analysis.json
"""

import json
import os

# ─────────────────────────────────────────────────────────────────────────────
# Feature Matrix
# ─────────────────────────────────────────────────────────────────────────────

SYSTEMS = ["ICDI-X", "Perplexity AI", "NotebookLM", "Plain RAG"]

# Feature: (dimension, description)
# Value per system: True (✓), False (✗), "partial" (~)
FEATURES = [
    # Core Retrieval
    ("Dense vector retrieval",              [True,  True,  True,  True]),
    ("Multi-hop graph reasoning",           [True,  False, False, False]),
    ("Information Bottleneck compression",  [True,  False, False, False]),
    ("Multi-Armed Bandit strategy select.", [True,  False, False, False]),
    ("Quantum-inspired similarity",         [True,  False, False, False]),

    # Documents & Modalities
    ("PDF document Q\\&A",                 [True,  "partial", True,  True]),
    ("Image / visual Q\\&A",               [True,  False, False, False]),
    ("Multi-document cross-synthesis",      [True,  False, True,  False]),
    ("Web-augmented retrieval",             [False, True,  False, False]),

    # Novel features
    ("Proactive insight generation",        [True,  False, "partial", False]),
    ("Bloom's taxonomy study guide",        [True,  False, False, False]),
    ("Cross-doc contradiction detection",   [True,  False, False, False]),
    ("Interactive knowledge graph",         [True,  False, False, False]),
    ("Audio overview (TTS)",               ["partial", False, True, False]),

    # UX
    ("Streaming SSE responses",             [True,  True,  False, False]),
    ("Multi-turn conversation memory",      [True,  True,  True,  False]),
    ("Exact page-level citations",          [True,  True,  True,  "partial"]),
    ("Confidence / uncertainty score",      [True,  "partial", False, False]),
    ("Input / output guardrails",           [True,  False, False, False]),

    # Evaluation
    ("Ablation study reported",             [True,  False, False, False]),
    ("Open-source / reproducible",         [True,  False, False, True]),
]

SYMBOL = {True: r"\ding{51}", False: r"\ding{55}", "partial": r"\textasciitilde"}
DISPLAY = {True: "✓", False: "✗", "partial": "~"}


def print_latex():
    col_spec = "|l|" + "c|" * len(SYSTEMS)
    header = " & ".join([r"\textbf{" + s + "}" for s in SYSTEMS])

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Feature Comparison: ICDI-X vs Existing Document AI Systems}",
        r"\label{tab:competitive}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        r"\textbf{Feature} & " + header + r" \\",
        r"\hline",
    ]

    sections = [
        ("Core Retrieval", slice(0, 5)),
        ("Documents \\& Modalities", slice(5, 9)),
        ("Novel Features", slice(9, 14)),
        ("UX \\& Safety", slice(14, 19)),
        ("Evaluation \\& Openness", slice(19, 21)),
    ]

    for sec_name, sl in sections:
        lines.append(r"\multicolumn{" + str(len(SYSTEMS)+1) + r"}{|l|}{\textit{\textbf{" + sec_name + r"}}} \\")
        lines.append(r"\hline")
        for feat_name, values in FEATURES[sl]:
            cells = " & ".join(SYMBOL[v] for v in values)
            lines.append(f"{feat_name} & {cells} \\\\")
        lines.append(r"\hline")

    lines += [
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\item \ding{51}~Supported \quad \ding{55}~Not supported \quad \textasciitilde~Partial/limited",
        r"\end{tablenotes}",
        r"\end{table}",
    ]

    print("\n".join(lines))


def print_plain():
    # Column widths
    feat_w = max(len(f[0]) for f in FEATURES) + 2
    col_w = 14

    header = "Feature".ljust(feat_w) + "".join(s.center(col_w) for s in SYSTEMS)
    print(header)
    print("-" * len(header))

    for feat_name, values in FEATURES:
        row = feat_name.ljust(feat_w) + "".join(DISPLAY[v].center(col_w) for v in values)
        print(row)


def save_json():
    os.makedirs("scripts/results", exist_ok=True)
    data = {
        "systems": SYSTEMS,
        "features": [
            {
                "feature": feat_name,
                "values": {sys: DISPLAY[val] for sys, val in zip(SYSTEMS, values)},
            }
            for feat_name, values in FEATURES
        ],
        "icdi_x_unique_features": [
            feat_name
            for feat_name, values in FEATURES
            if values[0] is True and all(v is not True for v in values[1:])
        ],
    }
    path = "scripts/results/competitive_analysis.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ JSON saved to {path}")
    print(f"   ICDI-X unique features: {len(data['icdi_x_unique_features'])}")
    for feat in data["icdi_x_unique_features"]:
        print(f"   ▸ {feat}")


if __name__ == "__main__":
    print("=" * 70)
    print("ICDI-X Competitive Analysis")
    print("=" * 70)
    print()
    print_plain()
    print()
    print("=" * 70)
    print("LaTeX Table:")
    print("=" * 70)
    print()
    print_latex()
    save_json()
