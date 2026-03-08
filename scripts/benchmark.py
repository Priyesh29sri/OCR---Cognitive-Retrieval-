"""
ICDI-X Benchmark — Paper-Quality Evaluation
============================================
Metrics:
  - BLEU-4            (nltk corpus_bleu, SmoothingFunction)
  - ROUGE-L           (rouge-score)
  - Keyword Recall    (fraction of ground-truth keywords in answer)
  - Faithfulness      (answer tokens supported by retrieved context)
  - Answer Relevancy  (answer tokens that match query keywords)
  - Expected Calibration Error (ECE) — confidence calibration
  - Latency (ms)      per pipeline stage via /query metadata

Usage:
  python scripts/benchmark.py --doc_id <id> --samples 50
  python scripts/benchmark.py --doc_id <id> --api http://127.0.0.1:8000 --samples 100
"""

import sys, os, json, time, argparse, statistics, ssl
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import nltk
for _pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

import requests
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from hotpotqa_dataset import get_dataset


API_BASE = "http://127.0.0.1:8000"
STOPWORDS = {"what", "is", "the", "how", "does", "a", "an", "of", "in", "to", "for", "and", "or", "this"}


def tokenize(text: str) -> list:
    return nltk.word_tokenize(text.lower())


def compute_bleu4(predictions: list, references: list) -> float:
    hyps = [tokenize(p) for p in predictions]
    refs = [[tokenize(r)] for r in references]
    sf = SmoothingFunction().method1
    return corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf)


def compute_rouge_l(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    p_scores, r_scores, f_scores = [], [], []
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        p_scores.append(result["rougeL"].precision)
        r_scores.append(result["rougeL"].recall)
        f_scores.append(result["rougeL"].fmeasure)
    return {
        "precision": statistics.mean(p_scores),
        "recall":    statistics.mean(r_scores),
        "f1":        statistics.mean(f_scores),
    }


def compute_keyword_recall(answer: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def compute_faithfulness(answer: str, context: str) -> float:
    if not context:
        return 0.0
    answer_tokens = set(tokenize(answer)) - STOPWORDS
    context_tokens = set(tokenize(context))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def compute_answer_relevancy(answer: str, query: str) -> float:
    query_tokens = set(tokenize(query)) - STOPWORDS
    answer_tokens = set(tokenize(answer))
    if not query_tokens:
        return 1.0
    return len(answer_tokens & query_tokens) / len(query_tokens)


def compute_f1_token(predicted: str, ground_truth: str) -> float:
    pred  = set(tokenize(predicted))  - STOPWORDS
    truth = set(tokenize(ground_truth)) - STOPWORDS
    if not pred or not truth:
        return 0.0
    inter = pred & truth
    p = len(inter) / len(pred)
    r = len(inter) / len(truth)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_ece(calibration_pairs: list, n_bins: int = 10) -> float:
    bins = [[] for _ in range(n_bins)]
    for conf, correct in calibration_pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))
    ece, n = 0.0, len(calibration_pairs)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(x[0] for x in b) / len(b)
        avg_acc  = sum(int(x[1]) for x in b) / len(b)
        ece += (len(b) / n) * abs(avg_conf - avg_acc)
    return ece


def query_backend(question: str, document_id: str, api_base: str, flags: dict = None) -> dict:
    payload = {"query": question, "document_id": document_id}
    if flags:
        payload.update(flags)
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{api_base}/query", json=payload, timeout=90)
        lat = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            d = resp.json()
            d["_latency_ms"] = lat
            return d
        return {"answer": "", "context": "", "_latency_ms": lat, "_error": resp.text, "metadata": {}}
    except Exception as e:
        lat = (time.perf_counter() - t0) * 1000
        return {"answer": "", "context": "", "_latency_ms": lat, "_error": str(e), "metadata": {}}


def run_benchmark(document_id: str, api_base: str = API_BASE, samples: int = 50,
                  flags: dict = None, label: str = "ICDI-X") -> dict:
    dataset = get_dataset()[:samples]
    print(f"\n{'='*72}")
    print(f"{label}  |  {len(dataset)} queries  |  doc_id={document_id}")
    print(f"{'='*72}")

    predictions, references = [], []
    kw_recalls, faithfulness_scores, relevancy_scores, f1_scores = [], [], [], []
    latencies, calibration_pairs = [], []
    errors = 0
    by_category: dict = {}

    for i, item in enumerate(dataset, 1):
        q, ref = item["query"], item["ground_truth"]
        kws     = item.get("keywords", [])
        cat     = item.get("category", "unknown")
        diff    = item.get("difficulty", "?")

        result  = query_backend(q, document_id, api_base, flags)
        answer  = result.get("answer", "")
        context = result.get("context", "")
        lat     = result.get("_latency_ms", 0)
        conf    = float(result.get("metadata", {}).get("confidence_score", 0.5))
        err     = result.get("_error")

        if err or not answer.strip():
            errors += 1
            print(f"  [{i:3d}] ⚠  {str(err or 'empty answer')[:60]}")
            continue

        f1  = compute_f1_token(answer, ref)
        kwr = compute_keyword_recall(answer, kws)
        fai = compute_faithfulness(answer, context)
        rel = compute_answer_relevancy(answer, q)

        predictions.append(answer)
        references.append(ref)
        kw_recalls.append(kwr)
        faithfulness_scores.append(fai)
        relevancy_scores.append(rel)
        f1_scores.append(f1)
        latencies.append(lat)
        calibration_pairs.append((conf, f1 > 0.3))

        by_category.setdefault(cat, {"f1": [], "kwr": []})
        by_category[cat]["f1"].append(f1)
        by_category[cat]["kwr"].append(kwr)

        print(f"  [{i:3d}] {cat:<12} {diff:<7} | F1={f1:.2f}  KWR={kwr:.2f}  Faith={fai:.2f}  Lat={lat:.0f}ms")

    if not predictions:
        print("  No successful predictions — check API + doc_id.")
        return {}

    bleu4   = compute_bleu4(predictions, references)
    rouge_l = compute_rouge_l(predictions, references)
    ece     = compute_ece(calibration_pairs)

    results = {
        "label":              label,
        "n_queries":          len(predictions),
        "n_errors":           errors,
        "bleu4":              round(bleu4, 4),
        "rouge_l_precision":  round(rouge_l["precision"], 4),
        "rouge_l_recall":     round(rouge_l["recall"], 4),
        "rouge_l_f1":         round(rouge_l["f1"], 4),
        "avg_f1_token":       round(statistics.mean(f1_scores), 4),
        "avg_keyword_recall": round(statistics.mean(kw_recalls), 4),
        "avg_faithfulness":   round(statistics.mean(faithfulness_scores), 4),
        "avg_answer_relevancy": round(statistics.mean(relevancy_scores), 4),
        "ece":                round(ece, 4),
        "avg_latency_ms":     round(statistics.mean(latencies), 1),
        "median_latency_ms":  round(statistics.median(latencies), 1),
        "p95_latency_ms":     round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        "by_category": {
            cat: {
                "avg_f1": round(statistics.mean(v["f1"]), 4),
                "avg_kwr": round(statistics.mean(v["kwr"]), 4),
                "n": len(v["f1"]),
            }
            for cat, v in by_category.items()
        },
    }

    print(f"\n{'─'*72}")
    print(f"  BLEU-4              : {results['bleu4']:.4f}")
    print(f"  ROUGE-L F1          : {results['rouge_l_f1']:.4f}")
    print(f"  Token F1            : {results['avg_f1_token']:.4f}")
    print(f"  Keyword Recall      : {results['avg_keyword_recall']:.4f}")
    print(f"  Faithfulness        : {results['avg_faithfulness']:.4f}")
    print(f"  Answer Relevancy    : {results['avg_answer_relevancy']:.4f}")
    print(f"  ECE (↓ better)      : {results['ece']:.4f}")
    print(f"  Avg Latency         : {results['avg_latency_ms']:.1f} ms")
    print(f"  P95 Latency         : {results['p95_latency_ms']:.1f} ms")
    print(f"  Errors              : {errors}/{len(dataset)}")
    print(f"{'='*72}")

    if by_category:
        print("\n  Per-category F1:")
        for cat, stats in results["by_category"].items():
            print(f"    {cat:<14} : F1={stats['avg_f1']:.3f}  KWR={stats['avg_kwr']:.3f}  (n={stats['n']})")

    return results


def print_latex_table(results_by_system: dict):
    print("\n% ── LaTeX Table ────────────────────────────────────────────────")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"System & BLEU-4 & ROUGE-L & Token F1 & KW-Recall & Faithfulness & Lat (ms) \\")
    print(r"\midrule")
    for name, r in results_by_system.items():
        if not r:
            continue
        print(
            f"{name} & {r.get('bleu4', 0):.3f} & {r.get('rouge_l_f1', 0):.3f} & "
            f"{r.get('avg_f1_token', 0):.3f} & {r.get('avg_keyword_recall', 0):.3f} & "
            f"{r.get('avg_faithfulness', 0):.3f} & {r.get('avg_latency_ms', 0):.0f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\caption{ICDI-X evaluation on 100-item multi-hop QA dataset.}")
    print(r"\label{tab:main_results}")
    print(r"\end{table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICDI-X Paper-Quality Benchmark")
    parser.add_argument("--api",    default=API_BASE)
    parser.add_argument("--doc_id", required=True, help="Document ID returned by /upload")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", default="scripts/benchmark_results.json")
    parser.add_argument("--label",  default="ICDI-X (Full)")
    args = parser.parse_args()

    results = run_benchmark(args.doc_id, args.api, args.samples, label=args.label)

    out_path = Path(__file__).parent.parent / args.output
    with open(out_path, "w") as f:
        json.dump({args.label: results}, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print_latex_table({args.label: results})
