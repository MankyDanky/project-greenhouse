"""
Re-measure a prepared RLHN parquet's token lengths with nanochat's tokenizer.

prepare_rlhn.py stored query_len / pos_lens / neg_lens with the Qwen3-0.6B tokenizer (a leftover of the
jina pipeline). The lengths only drive batch packing (the prompt builder re-tokenizes and truncates with
nanochat), but the estimates should come from the tokenizer the model uses. The tokenizer plays no other
part in data prep, so this gives exactly what prepare_rlhn.py --tokenizer <nanochat> would, without
re-downloading RLHN (whose Hub copy may have changed since).

Every other column is copied unchanged, row groups and row order included, and the output is checked
against the source column by column. Lengths are uncapped and measured with the prompt builder's own call:
encode(text, add_special_tokens=False, split_special_tokens=True).

  python relength_nanochat.py --src <prepared>/train.parquet --out <new dir> --tokenizer <nanochat HF dir>
"""
import argparse, datetime, json, os, time
from concurrent.futures import ProcessPoolExecutor   # raises BrokenProcessPool if a worker dies (a Pool hangs)

import numpy as np
import pyarrow as pa
import pyarrow.compute  # noqa: F401  (pa.compute)
import pyarrow.parquet as pq

LEN_COLS = ("query_len", "pos_lens", "neg_lens")
TOK = None


def _init(name):
    global TOK
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(name)


def _lens(texts, chunk=2000):
    # chunks: only the lengths are kept, never a row group's full token-id lists (those OOMed on RLHN-680K)
    out = []
    for i in range(0, len(texts), chunk):
        out += [len(x) for x in TOK(texts[i:i + chunk], add_special_tokens=False, split_special_tokens=True)["input_ids"]]
    return out


def _work(job):
    src, rg = job
    t = pq.ParquetFile(src).read_row_group(rg, columns=["query", "pos_texts", "neg_texts"])
    q = _lens(t.column("query").to_pylist())
    out = {"query_len": q}
    for col, name in (("pos_texts", "pos_lens"), ("neg_texts", "neg_lens")):
        lists = t.column(col).to_pylist()
        flat = _lens([s for l in lists for s in l])
        res, c = [], 0
        for l in lists:
            res.append(flat[c:c + len(l)]); c += len(l)
        out[name] = res
    return rg, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    pf = pq.ParquetFile(a.src)
    schema = pf.schema_arrow
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "train.parquet")
    tmp = dst + ".tmp"
    writer = pq.ParquetWriter(tmp, schema)
    qlen, plen, nlen, old_n = [], [], [], []   # lengths as numpy int32 chunks
    with ProcessPoolExecutor(a.procs, initializer=_init, initargs=(a.tokenizer,)) as pool:
        for rg, new in pool.map(_work, [(a.src, i) for i in range(pf.num_row_groups)]):
            t = pf.read_row_group(rg)
            for c in LEN_COLS:
                i = schema.get_field_index(c)
                old = t.column(c)
                if c != "query_len":   # same list sizes as the source: one length per passage
                    assert [len(x) for x in new[c]] == [len(x) for x in old.to_pylist()], (rg, c)
                if c == "neg_lens": old_n.append(np.asarray(pa.compute.list_flatten(old), dtype=np.int32))
                t = t.set_column(i, schema.field(i), pa.array(new[c], type=schema.field(i).type))
            writer.write_table(t)
            qlen.append(np.asarray(new["query_len"], dtype=np.int32))
            plen.append(np.asarray([x for l in new["pos_lens"] for x in l], dtype=np.int32))
            nlen.append(np.asarray([x for l in new["neg_lens"] for x in l], dtype=np.int32))
            if rg % 10 == 0: print(f"row group {rg + 1}/{pf.num_row_groups}  {time.time() - t0:.0f}s", flush=True)
    writer.close()

    # check: identical except the three length columns
    A, B = pq.ParquetFile(a.src), pq.ParquetFile(tmp)
    assert A.metadata.num_rows == B.metadata.num_rows and A.num_row_groups == B.num_row_groups
    assert A.schema_arrow.equals(B.schema_arrow, check_metadata=True)
    keep = [n for n in schema.names if n not in LEN_COLS]
    for rg in range(A.num_row_groups):
        assert A.read_row_group(rg, columns=keep).equals(B.read_row_group(rg, columns=keep)), rg
    os.replace(tmp, dst)

    q, p, n, o = (np.concatenate(x) for x in (qlen, plen, nlen, old_n))
    r = n / np.maximum(o, 1)
    pct = lambda x: {"mean": round(float(x.mean()), 1), "p50": int(np.percentile(x, 50)),
                     "p99": int(np.percentile(x, 99)), "max": int(x.max()), "over_512": round(float((x > 512).mean()), 4)}
    stats = {"rows": int(A.metadata.num_rows), "source": os.path.abspath(a.src),
             "source_mtime": datetime.datetime.fromtimestamp(os.path.getmtime(a.src)).isoformat(timespec="seconds"),
             "tokenizer": os.path.abspath(a.tokenizer) if os.path.isdir(a.tokenizer) else a.tokenizer,
             "call": "encode(text, add_special_tokens=False, split_special_tokens=True)",
             "changed_columns": list(LEN_COLS), "unchanged_columns": keep,
             "query_tokens": pct(q), "positive_tokens": pct(p), "negative_tokens": pct(n),
             "negative_nanochat_over_qwen": {"median": round(float(np.median(r)), 3),
                                             "p1": round(float(np.percentile(r, 1)), 3),
                                             "p99": round(float(np.percentile(r, 99)), 3)},
             "seconds": round(time.time() - t0)}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
