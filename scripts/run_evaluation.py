"""
ICDI-X — Master Evaluation Runner
====================================
Single-command paper-ready evaluation:
  1. Uploads a test document to the running backend
  2. Runs full benchmark (BLEU-4, ROUGE-L, F1, Faithfulness, ECE, Latency)
  3. Runs ablation study (5 configurations)
  4. Saves all JSON results
  5. Generates paper figures (4 PDFs + PNGs)
  6. Prints LaTeX tables ready to paste into paper

Usage:
  python scripts/run_evaluation.py --doc <path/to/doc.pdf> --samples 50
  python scripts/run_evaluation.py --doc_id <existing_id> --samples 50
  python scripts/run_evaluation.py --doc_id <id> --ablation_samples 20 --benchmark_samples 50
"""

import sys, os, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import requests

from benchmark      import run_benchmark, print_latex_table
from ablation_study import run_ablation, print_ablation_latex
from generate_figures import generate_all
from hotpotqa_dataset import print_dataset_stats


API_BASE   = "http://127.0.0.1:8000"
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"


def check_backend(api_base: str) -> bool:
    try:
        r = requests.get(f"{api_base}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        try:
            r = requests.get(f"{api_base}/docs", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


def upload_document(doc_path: str, api_base: str) -> str:
    print(f"\n[1/5] Uploading {doc_path} ...")
    with open(doc_path, "rb") as f:
        content_type = "application/pdf" if doc_path.endswith(".pdf") else "image/jpeg"
        resp = requests.post(
            f"{api_base}/upload",
            files={"file": (Path(doc_path).name, f, content_type)},
            timeout=120,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Upload failed: {resp.text}")
    doc_id = resp.json()["document_id"]
    print(f"    Uploaded. document_id = {doc_id}")
    print("    Waiting 15s for embedding background tasks ...")
    time.sleep(15)
    return doc_id


def main():
    parser = argparse.ArgumentParser(description="ICDI-X Master Evaluation Runner")
    parser.add_argument("--api",               default=API_BASE)
    parser.add_argument("--doc",               default=None,  help="Path to document to upload")
    parser.add_argument("--doc_id",            default=None,  help="Existing document_id (skip upload)")
    parser.add_argument("--benchmark_samples", type=int, default=50)
    parser.add_argument("--ablation_samples",  type=int, default=25)
    parser.add_argument("--skip_ablation",     action="store_true")
    parser.add_argument("--skip_figures",      action="store_true")
    parser.add_argument("--mab_state",         default="/tmp/mab_state.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  ICDI-X PAPER EVALUATION SUITE")
    print("=" * 72)

    print("\n[Dataset Statistics]")
    print_dataset_stats()

    if not check_backend(args.api):
        print(f"\n  ⚠  Backend not reachable at {args.api}")
        print("     Start it with:  bash start.sh  or  uvicorn app.main:app --port 8000")
        sys.exit(1)
    print(f"\n  ✓ Backend online at {args.api}")

    doc_id = args.doc_id
    if not doc_id:
        if not args.doc:
            print("\n  ERROR: Provide --doc <path> or --doc_id <id>")
            sys.exit(1)
        doc_id = upload_document(args.doc, args.api)

    benchmark_results_path = RESULTS_DIR / "benchmark_results.json"
    ablation_results_path  = RESULTS_DIR / "ablation_results.json"

    print(f"\n[2/5] Running benchmark ({args.benchmark_samples} queries) ...")
    bench = run_benchmark(
        document_id=doc_id,
        api_base=args.api,
        samples=args.benchmark_samples,
        label="ICDI-X (Full)",
    )
    with open(benchmark_results_path, "w") as f:
        json.dump({"ICDI-X (Full)": bench}, f, indent=2)
    print(f"    Saved → {benchmark_results_path}")

    ablation = {}
    if not args.skip_ablation:
        print(f"\n[3/5] Running ablation study ({args.ablation_samples} queries × 5 configs) ...")
        ablation = run_ablation(
            document_id=doc_id,
            api_base=args.api,
            samples=args.ablation_samples,
        )
        with open(ablation_results_path, "w") as f:
            json.dump(ablation, f, indent=2)
        print(f"    Saved → {ablation_results_path}")
    else:
        if ablation_results_path.exists():
            with open(ablation_results_path) as f:
                ablation = json.load(f)
            print(f"\n[3/5] Loaded existing ablation results from {ablation_results_path}")
        else:
            print("\n[3/5] Skipping ablation (--skip_ablation).")

    if not args.skip_figures:
        print(f"\n[4/5] Generating figures → {FIGURES_DIR}/")
        generate_all(
            ablation_path  = str(ablation_results_path) if ablation_results_path.exists() else None,
            benchmark_path = str(benchmark_results_path),
            mab_path       = args.mab_state,
            outdir         = str(FIGURES_DIR),
        )
    else:
        print("\n[4/5] Skipping figure generation (--skip_figures).")

    print(f"\n[5/5] LaTeX Tables\n")
    print_latex_table({"ICDI-X (Full)": bench})
    if ablation:
        print_ablation_latex(ablation)

    print("\n" + "=" * 72)
    print("  EVALUATION COMPLETE")
    print(f"  Results : {RESULTS_DIR}/")
    print(f"  Figures : {FIGURES_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
