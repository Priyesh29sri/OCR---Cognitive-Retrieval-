"""
ICDI-X Ablation Study — Paper-Ready Component Analysis
=======================================================
Tests 6 configurations by calling the live backend with component flags:

  Config 0 — Vanilla RAG    : dense only, no IB, no MAB, no graph, no guardrails
  Config 1 — +IB Filter     : dense + Information Bottleneck compression
  Config 2 — +MAB           : dense + IB + Multi-Armed Bandit selection
  Config 3 — +Graph         : dense + IB + MAB + Knowledge Graph reasoning
  Config 4 — +Quantum       : dense + IB + MAB + graph + quantum reranking
  Config 5 — Full ICDI-X    : all components enabled (default production)

Metrics per config: BLEU-4, ROUGE-L, Token F1, Keyword Recall, Faithfulness, Latency

Usage:
  python scripts/ablation_study.py --doc_id <id> --samples 30
"""

import sys, json, argparse, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import nltk
for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from benchmark import (
    run_benchmark, compute_bleu4, compute_rouge_l,
    compute_f1_token, compute_keyword_recall, print_latex_table
)
from hotpotqa_dataset import get_dataset


API_BASE = "http://127.0.0.1:8000"

CONFIGURATIONS = [
    {
        "name":  "Vanilla RAG",
        "short": "vanilla",
        "flags": {
            "use_ib_filtering":   False,
            "use_mab":            False,
            "use_graph_reasoning":False,
            "use_quantum":        False,
        },
        "description": "Dense vector retrieval only — no IB, no MAB, no graph"
    },
    {
        "name":  "+IB Filter",
        "short": "ib",
        "flags": {
            "use_ib_filtering":   True,
            "use_mab":            False,
            "use_graph_reasoning":False,
            "use_quantum":        False,
        },
        "description": "Dense + Information Bottleneck compression"
    },
    {
        "name":  "+MAB",
        "short": "mab",
        "flags": {
            "use_ib_filtering":   True,
            "use_mab":            True,
            "use_graph_reasoning":False,
            "use_quantum":        False,
        },
        "description": "Dense + IB + Multi-Armed Bandit arm selection"
    },
    {
        "name":  "+Graph",
        "short": "graph",
        "flags": {
            "use_ib_filtering":   True,
            "use_mab":            True,
            "use_graph_reasoning":True,
            "use_quantum":        False,
        },
        "description": "Dense + IB + MAB + Knowledge Graph multi-hop reasoning"
    },
    {
        "name":  "Full ICDI-X",
        "short": "full",
        "flags": {
            "use_ib_filtering":   True,
            "use_mab":            True,
            "use_graph_reasoning":True,
            "use_quantum":        True,
        },
        "description": "All components: dense + IB + MAB + graph + quantum reranking"
    },
]


def compute_delta(baseline: dict, current: dict, key: str) -> str:
    b = baseline.get(key, 0)
    c = current.get(key, 0)
    if b == 0:
        return "—"
    delta = c - b
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def run_ablation(document_id: str, api_base: str, samples: int) -> dict:
    all_results = {}

    for config in CONFIGURATIONS:
        print(f"\n{'█'*72}")
        print(f"  CONFIG: {config['name']}  |  {config['description']}")
        print(f"{'█'*72}")

        result = run_benchmark(
            document_id=document_id,
            api_base=api_base,
            samples=samples,
            flags=config["flags"],
            label=config["name"],
        )
        all_results[config["name"]] = result

    print_ablation_table(all_results)
    return all_results


def print_ablation_table(results: dict):
    configs = list(results.keys())
    baseline_name = configs[0] if configs else None
    baseline = results.get(baseline_name, {})

    print(f"\n{'='*90}")
    print("ABLATION STUDY — COMPONENT CONTRIBUTION TABLE")
    print(f"{'='*90}")

    header = f"{'Config':<18} {'BLEU-4':>8} {'ROUGE-L':>8} {'F1':>8} {'KWR':>8} {'Faith':>8} {'Lat(ms)':>9}"
    print(header)
    print("-" * 90)

    for name, r in results.items():
        if not r:
            print(f"  {name:<16}  (no results)")
            continue
        marker = "◀ baseline" if name == baseline_name else ""
        print(
            f"  {name:<16} "
            f"{r.get('bleu4', 0):>8.4f} "
            f"{r.get('rouge_l_f1', 0):>8.4f} "
            f"{r.get('avg_f1_token', 0):>8.4f} "
            f"{r.get('avg_keyword_recall', 0):>8.4f} "
            f"{r.get('avg_faithfulness', 0):>8.4f} "
            f"{r.get('avg_latency_ms', 0):>9.1f}  {marker}"
        )

    print("-" * 90)
    if baseline and len(results) > 1:
        full = results.get("Full ICDI-X", {})
        if full:
            bleu_gain = (full.get("bleu4", 0) - baseline.get("bleu4", 0))
            f1_gain   = (full.get("avg_f1_token", 0) - baseline.get("avg_f1_token", 0))
            print(f"\n  Full ICDI-X vs Vanilla RAG:")
            print(f"    BLEU-4 gain   : {bleu_gain:+.4f}")
            print(f"    Token F1 gain : {f1_gain:+.4f}")

    print(f"{'='*90}")


def print_ablation_latex(results: dict):
    print("\n% ── Ablation LaTeX Table ─────────────────────────────────────────")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Configuration & BLEU-4 & ROUGE-L & Token F1 & KW-Recall & Faithfulness & Lat (ms) \\")
    print(r"\midrule")
    for name, r in results.items():
        if not r:
            continue
        bold_open  = r"\textbf{" if name == "Full ICDI-X" else ""
        bold_close = "}"          if name == "Full ICDI-X" else ""
        print(
            f"{bold_open}{name}{bold_close} & "
            f"{r.get('bleu4',0):.3f} & {r.get('rouge_l_f1',0):.3f} & "
            f"{r.get('avg_f1_token',0):.3f} & {r.get('avg_keyword_recall',0):.3f} & "
            f"{r.get('avg_faithfulness',0):.3f} & {r.get('avg_latency_ms',0):.0f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\caption{Ablation study: contribution of each ICDI-X component.}")
    print(r"\label{tab:ablation}")
    print(r"\end{table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICDI-X Ablation Study")
    parser.add_argument("--api",     default=API_BASE)
    parser.add_argument("--doc_id",  required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output",  default="scripts/ablation_results.json")
    args = parser.parse_args()

    results = run_ablation(args.doc_id, args.api, args.samples)

    out_path = Path(__file__).parent.parent / args.output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print_ablation_latex(results)
