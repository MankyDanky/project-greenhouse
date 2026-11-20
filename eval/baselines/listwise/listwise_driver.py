"""
Rerank the eval datasets with rank_llm, one dataset per CLI process
(rank_llm/src/rank_llm/scripts/run_rank_llm.py, unmodified).

Why a driver instead of calling the CLI directly (measured 2026-09-10):
  * rank_llm + vLLM 0.29 at tensor-parallel 4 writes the TREC run and then never
    exits -- the async engine's teardown hangs with 4 GPUs held. The driver watches
    for the finished run file and then kills that process's whole session (engine +
    TP workers), so nothing is orphaned.
  * reusing one engine across datasets in-process does NOT work: rank_llm's async
    engine is bound to the first call's event loop and the second dataset gets
    EngineDeadError immediately -- while rank_llm still writes an output file. So
    every run is also VALIDATED (no EngineDead/Traceback in its log, every query of
    the requests file present) before a result is recorded.
Model reload per dataset is ~3 min once vLLM's compile cache is warm.
"""
import shutil, argparse, json, os, signal, subprocess, sys, time

QRELS = {"dl19": "qrels.dl19-passage.txt", "dl20": "qrels.dl20-passage.txt",
         "dl21": "qrels.dl21-passage.txt", "dl22": "qrels.dl22-passage.txt",
         "dl23": "qrels.dl23-passage.txt", "covid": "qrels.beir-v1.0.0-trec-covid.test.txt",
         "news": "qrels.beir-v1.0.0-trec-news.test.txt", "robust04": "qrels.beir-v1.0.0-robust04.test.txt",
         "nfc": "qrels.beir-v1.0.0-nfcorpus.test.txt", "scifact": "qrels.beir-v1.0.0-scifact.test.txt",
         "scidocs": "qrels.beir-v1.0.0-scidocs-test.txt", "fiqa": "qrels.beir-v1.0.0-fiqa-test.txt"}
BAD = ("EngineDeadError", "Traceback (most recent call last)", "CUDA out of memory")
BENIGN = ("import_utils",)   # vLLM logs a traceback for an optional module (deep_gemm) it cannot import


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def kill_session(p):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try: os.killpg(p.pid, sig)
        except ProcessLookupError: return
        for _ in range(20):
            if p.poll() is not None: break
            time.sleep(1)


def run_one(ds, a, req, run, jsonl, plog, timeout_s):
    argv = [sys.executable, a.launcher or "src/rank_llm/scripts/run_rank_llm.py", "--model_path", a.model,
            "--requests_file", req, "--top_k_candidates", str(a.top_k), "--window_size", str(a.window_size),
            "--stride", str(a.stride), "--context_size", str(a.context_size), "--variable_passages",
            "--max_passage_words", str(a.max_passage_words),
            "--num_gpus", str(a.num_gpus), "--prompt_template_path", a.template,
            "--output_trec_file", run + ".tmp", "--output_jsonl_file", jsonl]
    if a.thinking:
        argv += ["--is_thinking", "--reasoning_token_budget", str(a.reasoning_budget)]
    if a.max_queries > 0: argv += ["--max_queries", str(a.max_queries)]
    for f in (run + ".tmp", jsonl):
        if os.path.exists(f): os.remove(f)
    with open(plog, "w") as lf:
        p = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    t0, last_size, stable = time.time(), -1, 0
    while True:
        time.sleep(10)
        if p.poll() is not None:
            break                                   # exited on its own (success or failure)
        if os.path.exists(run + ".tmp"):
            s = os.path.getsize(run + ".tmp")
            stable = stable + 1 if s == last_size and s > 0 else 0
            last_size = s
            if stable >= 3:                         # written and unchanged for 30 s: done
                log(f"{ds}: run file complete; killing the hung engine session")
                kill_session(p); break
        if time.time() - t0 > timeout_s:
            log(f"{ds}: TIMEOUT after {timeout_s}s"); kill_session(p); return False, "timeout"
    bad = sorted({b for line in open(plog, errors="replace")
                  if not any(x in line for x in BENIGN) for b in BAD if b in line})
    if bad: return False, f"log contains {bad}"
    if not os.path.exists(run + ".tmp"): return False, f"no run file (rc={p.returncode})"
    want = sum(1 for _ in open(req))
    if a.max_queries > 0: want = min(want, a.max_queries)
    got = len({l.split()[0] for l in open(run + ".tmp")})
    if got != want: return False, f"run has {got} queries, requests file has {want}"
    os.replace(run + ".tmp", run)
    return True, f"{got} queries in {time.time() - t0:.0f}s"


def evaluate(ds, a, run, res_path, secs):
    E = a.eval_dir
    qrels = a.qrels_file or f"{E}/qrels/{QRELS[ds]}"
    ev = subprocess.run([os.environ.get("EVAL_TREC_EVAL") or shutil.which("trec_eval") or f"{E}/trec_eval", "-c", "-m", "ndcg_cut.10", "-m", "map", "-m", "recip_rank",
                         qrels, run], capture_output=True, text=True)
    m = {p[0]: float(p[2]) for p in (l.split() for l in ev.stdout.splitlines()) if len(p) == 3}
    res = {"dataset": ds, "backend": "rank_llm-listwise", "model": a.model, "tag": a.tag,
           "window": a.window_size, "stride": a.stride, "context": a.context_size,
           "n_queries": len({l.split()[0] for l in open(run)}), "seconds": int(secs), **m}
    with open(res_path, "w") as f: json.dump(res, f, indent=2)
    log("RESULT " + json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--eval-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results-dir", required=True); ap.add_argument("--num-gpus", type=int, required=True)
    ap.add_argument("--max-queries", type=int, default=-1)
    # apples-to-apples with the pointwise table: passages pre-truncated to 512 tokens of
    # the reranker's own tokenizer (pretruncate_requests.py), a word cap that never
    # binds, and a context that fits 20 x 512-token passages without rank_llm trimming
    ap.add_argument("--requests-dir", default=None, help="default: <eval-dir>/bm25")
    ap.add_argument("--context-size", type=int, default=16384)
    ap.add_argument("--max-passage-words", type=int, default=512)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--reasoning-budget", type=int, default=16384)
    ap.add_argument("--timeout-hours", type=float, default=12)
    ap.add_argument("--template", default=None)
    ap.add_argument("--window-size", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=100, help="candidates reranked per query (labeling: all)")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--qrels-file", default=None, help="override QRELS[dataset] (dev / training-group checks)")
    ap.add_argument("--launcher", default=None,
                    help="drop-in for run_rank_llm.py, e.g. a wrapper that sets vLLM pipeline parallelism")
    a = ap.parse_args()
    a.template = a.template or ("src/rank_llm/rerank/prompt_templates/qwen_thinking_template.yaml" if a.thinking
                                else "src/rank_llm/rerank/prompt_templates/qwen_non_thinking_template.yaml")
    log(f"settings: context {a.context_size}, max_passage_words {a.max_passage_words}, thinking={a.thinking}"
        f"{f' (budget {a.reasoning_budget})' if a.thinking else ''}, template {os.path.basename(a.template)}, "
        f"requests {a.requests_dir or 'bm25'}, launcher {a.launcher or 'run_rank_llm.py'}")
    failed = []
    for ds in a.datasets:
        res_path = os.path.join(a.results_dir, f"{a.tag}_{ds}.json")
        if os.path.exists(res_path):
            log(f"skip {ds} (done)"); continue
        req = os.path.join(a.requests_dir or os.path.join(a.eval_dir, "bm25"), f"retrieve_results_{ds}_top100.jsonl")
        run = os.path.join(a.out_dir, f"{ds}.trec")
        log(f"=== {ds}")
        t0 = time.time()
        ok, why = run_one(ds, a, req, run, os.path.join(a.out_dir, f"{ds}.jsonl"),
                          os.path.join(a.out_dir, f"{ds}.rank_llm.log"), int(a.timeout_hours * 3600))
        log(f"{ds}: {'OK' if ok else 'FAILED'} -- {why}")
        if ok: evaluate(ds, a, run, res_path, time.time() - t0)
        else: failed.append(ds)
    log(f"=== all done; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
