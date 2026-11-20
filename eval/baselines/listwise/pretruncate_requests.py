"""
Pre-truncate rank_llm request files so a listwise LLM sees the SAME document text as
the pointwise rerankers in evaluation/RESULTS.md: evaluation/src/rerank_eval.py
doc_text() (title + "\\n" + text, whitespace-normalised), cut at 512 tokens of the
reranker's OWN tokenizer -- exactly the rule every pointwise backend follows.

The cut uses the tokenizer's character offsets, so the kept text is an exact prefix
of the original (no decode round-trip). The candidate doc becomes {"text": ...} so
rank_llm renders it verbatim (no "Title: ... Content: ..." wrapper). rank_llm's own
--max_passage_words must then be >= 512 so it never cuts further (a whitespace word
is at least one token, so 512 words never binds on a 512-token text).
"""
import argparse, json, os, sys



def doc_text(cand):                      # verbatim copy of rerank_eval.doc_text (no max_chars)
    d = cand.get("doc") or {}
    t = d.get("contents") or d.get("text") or ""
    title = d.get("title") or ""
    if title and not t.startswith(title):
        t = f"{title}\n{t}"
    return " ".join(t.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--datasets", nargs="+", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    os.makedirs(a.dst, exist_ok=True)
    for ds in a.datasets:
        src = os.path.join(a.src, f"retrieve_results_{ds}_top100.jsonl")
        dst = os.path.join(a.dst, f"retrieve_results_{ds}_top100.jsonl")
        n_docs = n_cut = 0
        with open(src) as fi, open(dst + ".tmp", "w") as fo:
            for line in fi:
                r = json.loads(line)
                texts = [doc_text(c) for c in r["candidates"]]
                if not texts:     # NFCorpus has queries with no candidates: nothing to rerank;
                    continue      # trec_eval -c scores them 0 exactly as for every other reranker
                enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
                for c, t, offs in zip(r["candidates"], texts, enc["offset_mapping"]):
                    n_docs += 1
                    if len(offs) > a.max_tokens:
                        t = t[: offs[a.max_tokens - 1][1]]; n_cut += 1
                    c["doc"] = {"text": t}
                fo.write(json.dumps(r) + "\n")
        os.replace(dst + ".tmp", dst)
        print(f"{ds}: {n_docs} docs, {n_cut} cut at {a.max_tokens} tokens ({100 * n_cut / max(1, n_docs):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
