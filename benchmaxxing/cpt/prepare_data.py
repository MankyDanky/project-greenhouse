"""Build a continued-pretraining (CPT) data mix: raw corpus documents plus replayed pre-training text, written as
nanochat base-data shards. Everything is set by one YAML file, configs/data.yaml; any key can be overridden on the
command line (--set KEY=VALUE, dotted keys, YAML values).

    python prepare_data.py configs/data.yaml                                   # the mix, data/ir_mix
    python prepare_data.py configs/data.yaml --set replay.fraction=0.1 name=ir_mix_r10  # any key can be overridden
    python prepare_data.py configs/data.yaml --stage corpus                    # only (re)build the per-source caches
    python prepare_data.py configs/data.yaml --limit 2000                      # smoke test: at most 2,000 docs per source

The mix is written to data/<name> unless `out_dir` is set; manifest.json records the configuration with overrides.

Stages
  corpus  For each enabled source: read the documents, apply `sample` if set (a fixed fraction by document id),
          format them as `title\\n\\ntext` (or the text alone), split documents longer than `format.max_chars` at a
          paragraph / sentence / word boundary, drop exact duplicates, set aside a held-out sample (by hash), count
          tokens. Cached per source in `work_dir/corpus/`, in a fixed pseudo-random order, so any later mix reuses it.
          Memory stays bounded for any corpus size.
  replay  Read pre-training documents (ClimbMix shards from the Hugging Face hub) in a fixed, seeded order until the
          replay token target is met. A mix with a smaller replay target uses a prefix of the same stream.
  mix     Interleave the corpus sources in proportion to their tokens (a source with several passes runs them one
          after another, each over an equal part of the run), then corpus and replay so that the cumulative replay share follows `replay.schedule`, and write
          `out_dir/train/shard_*.parquet` (column `text`), the evaluation sets, and `out_dir/manifest.json`.

Output layout (`out_dir`)
  train/shard_00000.parquet ...   training shards; the LAST one is the held-out mix nanochat reports as val bpb
  heldout_corpus/                  corpus pieces never trained on (gain on the domain)
  seen_corpus/                     as many trained-on corpus pieces, per source (gain minus this = memorisation)
  heldout_replay/                  documents of the pre-training validation shard (forgetting)
                                   (each two shards, the layout scripts/base_eval.py --eval bpb reads)
  manifest.json                    sources, token counts, realised replay share per shard, SHA256 of every shard

What is never used: queries, relevance judgements, or any selection by query. Only document text.
"""
import argparse, gzip, hashlib, heapq, json, math, os, random, shutil, sys, time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfgload import load_config  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def h64(s):
    """Stable 64-bit hash of a string (whitespace- and case-insensitive), used for dedup, held-out and sampling."""
    return int.from_bytes(hashlib.blake2b(" ".join(s.lower().split()).encode("utf-8"), digest_size=8).digest(), "little")


def unit(x, salt):
    """Map a 64-bit hash to [0, 1), decorrelated by `salt` (so held-out and sampling draws are independent)."""
    return int.from_bytes(hashlib.blake2b(f"{salt}:{x}".encode(), digest_size=8).digest(), "little") / 2 ** 64


def pieces(s, max_chars):
    """Split a long document at a paragraph break, else a sentence end, else a word boundary, in the second half of
    each `max_chars` window. Short documents are returned unchanged."""
    if len(s) <= max_chars:
        return [s]
    out = []
    while len(s) > max_chars:
        win = s[:max_chars]
        cut = win.rfind("\n", max_chars // 2)
        if cut < 0:
            cut = win.rfind(". ", max_chars // 2)
            cut = cut + 1 if cut >= 0 else -1
        if cut < 0:
            cut = win.rfind(" ", max_chars // 2)
        if cut <= 0:
            cut = max_chars
        out.append(s[:cut].strip()); s = s[cut:].strip()
    if s:
        out.append(s)
    return [p for p in out if p]


# ---------------------------------------------------------------------------------------------------------- sources
def read_source(src, limit=0):
    """Yield (doc_id, title, text) for one source spec. Types:
      hf_parquet  {repo, file, id, title, text}        a parquet file in a Hugging Face dataset repo (e.g. BeIR/<name>)
      hf_jsonl    {repo, file, id, title, text}        a (gzipped) JSON-lines file in a Hugging Face dataset repo
      jsonl       {path, id, title, text}              a local (gzipped) JSON-lines file, e.g. a licensed collection
      ir_datasets {id}                                 any ir_datasets corpus (doc_id, title if present, text)
      pyserini_index {index, id, title, text}          every document stored in a Pyserini prebuilt index (raw JSON)
    """
    n = 0
    t = src["type"]
    if t in ("hf_parquet", "hf_jsonl"):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(src["repo"], src["file"], repo_type="dataset", revision=src.get("revision"))
    else:
        path = src.get("path")
    if t == "hf_parquet":
        pf = pq.ParquetFile(path)
        cols = [c for c in (src.get("id"), src.get("title"), src["text"]) if c]
        for batch in pf.iter_batches(columns=cols, batch_size=4096):
            d = batch.to_pydict()
            for i in range(len(d[src["text"]])):
                yield (str(d[src["id"]][i]) if src.get("id") else str(n),
                       (d[src["title"]][i] or "") if src.get("title") else "", d[src["text"]][i] or "")
                n += 1
                if limit and n >= limit:
                    return
    elif t in ("hf_jsonl", "jsonl"):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                yield (str(d.get(src.get("id", "_id"), n)), d.get(src.get("title", "title")) or "",
                       d.get(src.get("text", "text")) or "")
                n += 1
                if limit and n >= limit:
                    return
    elif t == "pyserini_index":
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher.from_prebuilt_index(src["index"])
        for i in range(searcher.num_docs):
            doc = searcher.doc(i)
            raw = (doc.raw() or doc.contents() or "").strip()
            d = json.loads(raw) if raw.startswith("{") else {src.get("text", "text"): raw}
            yield (str(d.get(src.get("id", "_id"), doc.docid())), d.get(src.get("title", "title")) or "",
                   d.get(src.get("text", "text")) or "")
            n += 1
            if limit and n >= limit:
                return
    elif t == "ir_datasets":
        import ir_datasets
        for doc in ir_datasets.load(src["id"]).docs_iter():
            yield doc.doc_id, getattr(doc, "title", "") or "", getattr(doc, "text", "") or ""
            n += 1
            if limit and n >= limit:
                return
    else:
        raise ValueError(f"unknown source type {t!r}")


class Counter:
    """Token counts with the backbone's tokenizer (a `tokenizer.json`), plus one BOS per document as nanochat's
    loader prepends it."""
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(path))

    def __call__(self, texts):
        return [len(e.ids) + 1 for e in self.tok.encode_batch(texts, add_special_tokens=False)]


# ----------------------------------------------------------------------------------------------------------- stages
N_BUCKETS = 256


def build_corpus_source(spec, cfg, counter, work, limit):
    """Cache one source as work_dir/corpus/<name>.arrow (text, n_tokens), its held-out rows as <name>.heldout.npy and
    its statistics as <name>.json. Memory stays bounded whatever the corpus size (MS MARCO v2 has 138M passages):
    pieces are spread over N_BUCKETS files by content hash, so exact duplicates share a bucket, and each bucket is
    deduplicated (the first occurrence is kept), ordered by hash and token-counted on its own. The cache is therefore
    in a fixed pseudo-random order that depends only on the text, the same for every mix and seed."""
    name = spec["name"]
    d = work / "corpus"
    out, meta_path, held_path = d / f"{name}.arrow", d / f"{name}.json", d / f"{name}.heldout.npy"
    if out.exists() and meta_path.exists() and held_path.exists() and not limit:
        log(f"corpus {name}: cached ({out})")
        return
    max_chars = cfg["format"]["max_chars"]; sep = cfg["format"].get("title_separator", "\n\n")
    ho = cfg["heldout"]; sample = float(spec.get("sample", 1.0))
    stats = {"read": 0, "duplicates": 0, "sampled_out": 0}

    # pass 1: read, format, split, and spread the pieces over hash buckets
    bdir = d / f"{name}.buckets"
    if bdir.exists():
        shutil.rmtree(bdir)
    bdir.mkdir(parents=True)
    bschema = pa.schema([("text", pa.string()), ("hash", pa.uint64()), ("seq", pa.int64())])
    writers, bufs, seq = {}, {}, 0

    def flush(b):
        rows = bufs.pop(b, None)
        if not rows:
            return
        if b not in writers:
            sink = pa.OSFile(str(bdir / f"{b:03d}.arrow"), "wb")
            writers[b] = (sink, pa.ipc.new_stream(sink, bschema))
        texts, hashes, seqs = zip(*rows)
        writers[b][1].write_batch(pa.record_batch([pa.array(texts, pa.string()), pa.array(hashes, pa.uint64()),
                                                   pa.array(seqs, pa.int64())], schema=bschema))

    for doc_id, title, text in read_source(spec["source"], limit):
        stats["read"] += 1
        if stats["read"] % 5_000_000 == 0:
            log(f"corpus {name}: {stats['read']:,} read")
        if sample < 1.0 and unit(h64(f"{name}:{doc_id}"), "sample") >= sample:
            stats["sampled_out"] += 1
            continue
        title, text = title.strip(), text.strip()
        full = f"{title}{sep}{text}" if (title and text) else (text or title)
        for piece in pieces(full, max_chars) if full else []:
            k = h64(piece); b = k % N_BUCKETS
            bufs.setdefault(b, []).append((piece, k, seq)); seq += 1
            if len(bufs[b]) >= 2048:
                flush(b)
    for b in list(bufs):
        flush(b)
    for sink, w in writers.values():
        w.close(); sink.close()

    # pass 2: per bucket, drop duplicates, order by hash, count tokens; collect held-out candidates
    schema = pa.schema([("text", pa.string()), ("n_tokens", pa.int32())])
    tmp = out.with_name(out.name + ".tmp")
    cand_u, cand_row, row = [], [], 0
    with pa.OSFile(str(tmp), "wb") as f, pa.ipc.new_file(f, schema) as w:
        for b in range(N_BUCKETS):
            bp = bdir / f"{b:03d}.arrow"
            if not bp.exists():
                continue
            with pa.OSFile(str(bp), "rb") as fh:          # read into memory and close: buckets are deleted below
                t = pa.ipc.open_stream(fh).read_all()
            k = t.column("hash").to_numpy(); order = np.lexsort((t.column("seq").to_numpy(), k))
            ks = k[order]
            first = np.ones(len(ks), bool); first[1:] = ks[1:] != ks[:-1]
            stats["duplicates"] += int((~first).sum())
            keep = order[first]
            texts = t.column("text").take(pa.array(keep)).to_pylist()
            for i in range(0, len(texts), 20000):
                chunk = texts[i:i + 20000]
                w.write_batch(pa.record_batch([pa.array(chunk, pa.string()), pa.array(counter(chunk), pa.int32())],
                                              schema=schema))
            # held-out draw: a uniform number per piece from its hash (a multiplicative hash, independent of the order)
            u = (ks[first] * np.uint64(0x9E3779B97F4A7C15)).astype(np.float64) / 2.0 ** 64
            sel = np.nonzero(u < float(ho["fraction"]))[0]
            cand_u.append(u[sel]); cand_row.append(row + sel)
            row += len(keep)
            del t, texts
    tmp.replace(out)
    assert bdir.name.endswith(".buckets"); shutil.rmtree(bdir)
    cu = np.concatenate(cand_u) if cand_u else np.zeros(0); cr = np.concatenate(cand_row) if cand_row else np.zeros(0, int)
    cap = int(ho["max_docs_per_source"])
    held_rows = np.sort(cr[np.argsort(cu, kind="stable")[:cap]]).astype(np.int64)   # at most `cap`: the smallest draws
    np.save(held_path, held_rows)
    ntok = np.asarray(open_arrow(out).column("n_tokens").to_numpy(), np.int64)
    held = np.zeros(len(ntok), bool); held[held_rows] = True
    stats.update(pieces=int(len(ntok)), heldout_docs=int(held.sum()), train_docs=int((~held).sum()),
                 train_tokens=int(ntok[~held].sum()), heldout_tokens=int(ntok[held].sum()))
    meta_path.write_text(json.dumps(stats, indent=2))
    log(f"corpus {name}: {stats['train_docs']:,} train docs / {stats['train_tokens']:,} tokens, "
        f"{stats['heldout_docs']:,} held out, {stats['duplicates']:,} duplicates dropped")


def open_arrow(path):
    return pa.ipc.open_file(pa.memory_map(str(path), "r")).read_all()


def replay_stream(cfg, counter, work, target_tokens, limit):
    """Pre-training documents in a fixed, seeded shard order until `target_tokens`, as an Arrow table (text, n_tokens).
    The stream is cached in work_dir/replay/ and memory-mapped; a mix with a smaller target reads a prefix of the
    same stream."""
    r = cfg["replay"]; src = r["source"]
    schema = pa.schema([("text", pa.string()), ("n_tokens", pa.int32())])
    if target_tokens <= 0:
        return schema.empty_table()
    key = h64(json.dumps([cfg["seed"], src, r["heldout_shard"], limit], sort_keys=True))
    path = work / "replay" / f"stream_{key:016x}.arrow"
    if path.exists():
        t = open_arrow(path)
        cum = np.cumsum(t.column("n_tokens").to_numpy())
        if len(cum) and cum[-1] >= target_tokens:
            k = int(np.searchsorted(cum, target_tokens)) + 1
            log(f"replay: {k:,} docs / {int(cum[k - 1]):,} tokens from the cached stream {path.name}")
            return t.slice(0, k)
        log(f"replay: cached stream has {int(cum[-1]) if len(cum) else 0:,} tokens; rebuilding to {target_tokens:,}")
    from huggingface_hub import hf_hub_download
    lo, hi = src["shards"]
    rng = random.Random(cfg["seed"])
    order = [i for i in range(lo, hi + 1) if i != r["heldout_shard"]]; rng.shuffle(order)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    total = ndocs = 0
    with pa.OSFile(str(tmp), "wb") as f, pa.ipc.new_file(f, schema) as w:
        for j, sid in enumerate(order):
            fp = hf_hub_download(src["repo"], src["file_pattern"].format(sid), repo_type="dataset")
            for batch in pq.ParquetFile(fp).iter_batches(columns=[src.get("text", "text")], batch_size=8192):
                chunk = [t for t in batch.column(0).to_pylist() if t]
                ntok = counter(chunk)
                cum = np.cumsum(ntok)
                k = len(chunk)
                if total + (cum[-1] if k else 0) >= target_tokens:
                    k = int(np.searchsorted(cum, target_tokens - total)) + 1
                if limit:
                    k = min(k, limit - ndocs)
                w.write_batch(pa.record_batch([pa.array(chunk[:k], pa.string()), pa.array(ntok[:k], pa.int32())],
                                              schema=schema))
                total += int(cum[k - 1]) if k else 0; ndocs += k
                if total >= target_tokens or (limit and ndocs >= limit):
                    break
            else:
                continue
            break
        else:
            raise RuntimeError(f"replay source exhausted at {total:,} of {target_tokens:,} tokens")
    tmp.replace(path)
    log(f"replay: {ndocs:,} docs / {total:,} tokens from {j + 1} shard(s), cached as {path.name}")
    return open_arrow(path)


def heldout_replay(cfg, counter, n_docs):
    from huggingface_hub import hf_hub_download
    r = cfg["replay"]; src = r["source"]
    path = hf_hub_download(src["repo"], src["file_pattern"].format(r["heldout_shard"]), repo_type="dataset")
    texts = []
    for batch in pq.ParquetFile(path).iter_batches(columns=[src.get("text", "text")], batch_size=4096):
        texts += [t for t in batch.column(0).to_pylist() if t]
        if len(texts) >= n_docs:
            break
    return texts[:n_docs]


def replay_share_target(schedule, skew, f, x):
    """Cumulative replay tokens wanted at run progress x in [0, 1], as a fraction of all tokens so far.
    constant: share f everywhere. front_loaded / back_loaded: the instantaneous share is f * (1 + s * (1 - 2x)) with
    s = +skew / -skew (clipped to stay within [0, 1]), so the run-level share is still f."""
    if schedule == "constant" or skew == 0:
        return f
    s = skew if schedule == "front_loaded" else -skew
    x = max(x, 1e-9)
    return f * (1 + s * (1 - x))        # average of f*(1 + s*(1 - 2t)) over t in [0, x]


def write_shards(docs, out_dir, prefix, tokens_per_shard, row_group_docs, manifest_key, manifest):
    """docs: iterable of (text, n_tokens, is_replay). Writes `prefix` shards of about `tokens_per_shard` tokens."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shard, buf, tok, rep = 0, [], 0, 0
    records = []

    def flush():
        nonlocal shard, buf, tok, rep
        if not buf:
            return
        path = out_dir / f"{prefix}{shard:05d}.parquet"
        pq.write_table(pa.table({"text": buf}), path, row_group_size=row_group_docs)
        records.append({"file": path.name, "docs": len(buf), "tokens": tok, "replay_share": rep / max(tok, 1)})
        shard += 1; buf, tok, rep = [], 0, 0

    for text, n, is_replay in docs:
        buf.append(text); tok += n; rep += n if is_replay else 0
        if tok >= tokens_per_shard:
            flush()
    flush()
    manifest[manifest_key] = records
    return records


class RowReader:
    """Random access to one row of a memory-mapped, multi-chunk Arrow column, without combining its chunks."""
    def __init__(self, col):
        self.chunks = col.chunks
        self.starts = np.cumsum([0] + [len(c) for c in self.chunks])

    def __getitem__(self, i):
        c = int(np.searchsorted(self.starts, i, side="right")) - 1
        return self.chunks[c][int(i - self.starts[c])].as_py()


def pass_segments(train_rows, ntok, passes, rng):
    """The rows a source contributes, pass by pass, for `passes` (any non-negative number) passes' worth of tokens:
    the first pass in cache order, so a partial first pass is a fixed random subset of the source (the cache is in
    hash order) and a smaller amount is always a subset of a larger one; later passes in seeded permutations. The
    last pass is cut where the token budget is reached. Returns (list of row arrays, tokens)."""
    budget = passes * float(ntok[train_rows].sum()) if len(train_rows) else 0.0
    segs, done, k = [], 0, 0
    while done < budget - 0.5:
        order = train_rows if k == 0 else rng.permutation(train_rows)
        cum = np.cumsum(ntok[order])
        if done + cum[-1] <= budget:
            segs.append(order); done += int(cum[-1])
        else:
            n = int(np.searchsorted(cum, budget - done)) + 1
            segs.append(order[:n]); done += int(cum[n - 1])
            break
        k += 1
    return segs, done


def source_stream(t, ntok, segs):
    """(text, n_tokens) for a source's segments: the first is read sequentially (rows ascending), the others by row."""
    col = t.column("text")
    if segs:
        first = segs[0]
        starts = np.cumsum([0] + [len(c) for c in col.chunks])
        for ci, c in enumerate(col.chunks):
            lo, hi = np.searchsorted(first, [starts[ci], starts[ci + 1]])
            if lo >= len(first):
                break
            if lo == hi:
                continue
            texts = c.to_pylist()
            for r in first[lo:hi]:
                yield texts[r - starts[ci]], int(ntok[r])
    rr = RowReader(col)
    for seg in segs[1:]:
        for r in seg:
            yield rr[r], int(ntok[r])


def plan_passes(cfg, available):
    """Passes per source. `available`: {source name: training tokens in one pass}.
    Every source has `passes`: the most passes it may get (default 1), and its weight within its `group` (default:
    the source's own name). Without `groups` in the config, every source gets its `passes`. With `groups`, each
    group g has a `ratio`: its share of the corpus tokens, relative to the other groups. A group's capacity is the
    sum of passes x tokens over its sources; the group that runs out first sets the corpus size, every other group
    gets its ratio of it, and the sources of a group are scaled together (so they keep their relative weights).
    `ratio: all` takes a group's full capacity outside that calculation. With `exclude_ood`, `groups_exclude_ood`
    (if given) replaces `groups`. Returns ({source: passes}, {group: info})."""
    specs = {sp["name"]: sp for sp in cfg["corpus"]}
    full = {n: float(specs[n].get("passes", 1)) for n in available}
    groups = cfg.get("groups_exclude_ood") if cfg.get("exclude_ood") and cfg.get("groups_exclude_ood") else cfg.get("groups")
    if not groups:
        return full, {}
    members = {}
    for n in available:
        members.setdefault(specs[n].get("group", n), []).append(n)
    unknown = set(members) - set(groups)
    assert not unknown, f"sources in groups that `groups` does not define: {sorted(unknown)}"
    cap = {g: sum(full[n] * available[n] for n in ns) for g, ns in members.items()}
    weighted = {g: float(groups[g]["ratio"]) for g in members if str(groups[g]["ratio"]) != "all"}
    wsum = sum(weighted.values())
    size = min((cap[g] * wsum / w for g, w in weighted.items() if w > 0), default=0.0)
    out, info = {}, {}
    for g, ns in members.items():
        if g in weighted:
            budget = weighted[g] / wsum * size if wsum else 0.0
        else:
            budget = cap[g]
        scale = budget / cap[g] if cap[g] else 0.0
        scale = 1.0 if abs(scale - 1) < 1e-9 else scale
        for n in ns:
            out[n] = full[n] * scale
        info[g] = {"ratio": groups[g]["ratio"], "capacity_tokens": int(cap[g]), "budget_tokens": int(budget),
                   "scale": round(scale, 4), "sets_the_size": bool(g in weighted and abs(scale - 1) < 1e-9)}
    return out, info


def interleave(streams, totals):
    """Merge the sources so that, at every point, each one's share of the corpus tokens so far equals its share of
    the whole (each document is placed at the midpoint of its token span within its source)."""
    def keyed(stream, total):
        done = 0
        for x, n in stream:
            yield (done + n / 2) / total, x, n
            done += n
    for _, x, n in heapq.merge(*(keyed(st, T) for st, T in zip(streams, totals) if T > 0), key=lambda e: e[0]):
        yield x, n


def table_stream(t):
    for tc, nc in zip(t.column("text").chunks, t.column("n_tokens").chunks):
        yield from zip(tc.to_pylist(), (int(n) for n in nc.to_numpy()))


def enabled_sources(cfg):
    """`corpus` entries that are enabled (`enabled`, default true, or named in `enable_sources`), minus those marked
    `ood: true` when `exclude_ood` is set (they are never read, and the mix is sized without them). Sources in
    `exclude_sources` are included here: an excluded source still counts when the mix is sized (see mix), so that
    every other source contributes exactly what it would without the exclusion."""
    names = {s["name"] for s in cfg["corpus"]}
    for key in ("exclude_sources", "enable_sources"):
        unknown = set(cfg.get(key) or []) - names
        assert not unknown, f"{key} names unknown sources: {sorted(unknown)}"
    forced = set(cfg.get("enable_sources") or [])
    on = [s for s in cfg["corpus"] if s.get("enabled", True) or s["name"] in forced]
    if cfg.get("exclude_ood"):          # out-of-domain collections stay out of the mix entirely (before it is sized)
        on = [s for s in on if not s.get("ood")]
    return on


def mix(cfg, work, limit):
    counter = Counter(HERE / cfg["tokenizer"])
    out = HERE / (cfg.get("out_dir") or f"data/{cfg['name']}")
    rng = np.random.default_rng(cfg["seed"])
    sources = enabled_sources(cfg)
    excluded = set(cfg.get("exclude_sources") or [])
    streams, totals, held_texts, seen_texts = [], [], [], []
    manifest = {"name": cfg["name"], "config": cfg, "sources": {}}
    loaded = {}
    for s in sources:
        t = open_arrow(work / "corpus" / f"{s['name']}.arrow")
        ntok = np.asarray(t.column("n_tokens").to_numpy(), np.int64)
        held = np.zeros(len(ntok), bool); held[np.load(work / "corpus" / f"{s['name']}.heldout.npy")] = True
        loaded[s["name"]] = (t, ntok, held, np.nonzero(~held)[0])
    passes, group_info = plan_passes(cfg, {n: int(v[1][v[3]].sum()) for n, v in loaded.items()})
    manifest["groups"] = group_info
    for g, gi in group_info.items():
        log(f"plan: group {g}: {gi['budget_tokens']:,} tokens of {gi['capacity_tokens']:,} available "
            f"(x{gi['scale']:.3f}){' - sets the corpus size' if gi['sets_the_size'] else ''}")
    for s in sources:
        if s["name"] in excluded:
            log(f"plan: {s['name']} excluded ({passes[s['name']]:.2f} passes would have been used)")
            continue
        t, ntok, held, idx = loaded[s["name"]]
        rng = np.random.default_rng([cfg["seed"], h64(s["name"]) % 2 ** 32])
        segs, tokens = pass_segments(idx, ntok, passes[s["name"]], rng)
        rr = RowReader(t.column("text"))
        held_idx = np.nonzero(held)[0]
        held_texts += [rr[i] for i in held_idx]
        # memorisation probe: as many TRAINED pieces of this source as it has held-out ones (from its first pass, i.e.
        # pieces that are in the mix), chosen independently of the run's seed and of the other sources
        used = segs[0] if segs else idx[:0]
        srng = np.random.default_rng(h64(f"seen:{s['name']}") % 2 ** 32)
        seen_idx = srng.choice(used, size=min(len(held_idx), len(used)), replace=False) if len(used) else used
        seen_texts += [rr[i] for i in np.sort(seen_idx)]
        streams.append(source_stream(t, ntok, segs))
        totals.append(tokens)
        manifest["sources"][s["name"]] = {**json.loads((work / "corpus" / f"{s['name']}.json").read_text()),
                                          "group": s.get("group", s["name"]), "passes": round(passes[s["name"]], 4),
                                          "tokens_in_mix": tokens}
        log(f"plan: {s['name']}: {passes[s['name']]:.3f} passes, {tokens:,} tokens")
    corpus_tokens = int(sum(totals))
    r = cfg["replay"]; f = float(r["fraction"])
    if r.get("tokens"):                 # an absolute replay budget, e.g. the replay-only control (no corpus at all)
        tok = r["tokens"]
        if isinstance(tok, str) and tok.startswith("from:"):     # as many tokens as another built mix
            tok = json.loads((HERE / tok[5:] / "manifest.json").read_text())["totals"]["train_tokens"]
        replay_target = int(tok)
        f = replay_target / (corpus_tokens + replay_target)
    else:
        assert f < 1, "replay.fraction must be < 1; for a replay-only run set replay.tokens and disable the corpus"
        replay_target = int(round(corpus_tokens * f / (1 - f))) if f > 0 else 0
    log(f"mix: {corpus_tokens:,} corpus tokens from {len(streams)} source(s); replay target {replay_target:,} "
        f"tokens ({f:.0%})")
    rep = replay_stream(cfg, counter, work, replay_target, limit)
    replay_tokens = int(np.asarray(rep.column("n_tokens").to_numpy(), np.int64).sum()) if rep.num_rows else 0
    total = corpus_tokens + replay_tokens

    def merged():
        """Take the next replay document whenever the replay share so far is below the schedule, else the next
        corpus document (the corpus sources themselves interleaved in proportion)."""
        corpus, replay = interleave(streams, totals), table_stream(rep)
        c, rp = next(corpus, None), next(replay, None)
        ctok = rtok = 0
        while c is not None or rp is not None:
            done = ctok + rtok
            want = replay_share_target(r.get("schedule", "constant"), float(r.get("skew", 0.0)), f, done / total)
            if rp is not None and (c is None or rtok < want * done or (done == 0 and f > 0)):
                yield rp[0], rp[1], True
                rtok += rp[1]; rp = next(replay, None)
            else:
                yield c[0], c[1], False
                ctok += c[1]; c = next(corpus, None)

    o = cfg["output"]
    train_dir = out / "train"
    for sub in ("train", "heldout_corpus", "seen_corpus", "heldout_replay"):   # a rebuilt mix keeps no stale shards
        for old in (out / sub).glob("*.parquet"):
            old.unlink()
    recs = write_shards(merged(), train_dir, "shard_", int(o["tokens_per_shard"]), int(o["row_group_docs"]), "train", manifest)
    # evaluation sets, two shards each (scripts/base_eval.py --eval bpb reports the first as "train", the second as
    # "val"): corpus pieces never trained on, the same number of corpus pieces that were trained on, and documents
    # from the pre-training validation shard
    n_rep = int(r.get("heldout_docs", 4000))
    rep_held = heldout_replay(cfg, counter, n_rep) if f > 0 or r.get("measure_forgetting", True) else []
    rng2 = random.Random(cfg["seed"] + 1)
    for name, texts in (("heldout_corpus", held_texts), ("seen_corpus", seen_texts), ("heldout_replay", rep_held)):
        texts = list(texts); rng2.shuffle(texts)
        half = len(texts) // 2
        parts = (texts[:half], texts[half:])
        if half:
            d = out / name; d.mkdir(parents=True, exist_ok=True)
            for k, part in enumerate(parts):     # small row groups, so every rank of a multi-GPU eval gets data
                pq.write_table(pa.table({"text": part}), d / f"shard_{k:05d}.parquet", row_group_size=16)
        manifest[name] = {"docs": len(texts), "tokens_per_shard": [sum(counter(part)) if part else 0 for part in parts]}
    # the last training shard is nanochat's in-training val set: the held-out mix at the run's replay share
    if held_texts:
        k_rep = int(round(len(held_texts) * f / max(1 - f, 1e-9))) if f > 0 else 0
    else:                               # replay-only control: no corpus, so the val set is held-out replay alone
        k_rep = len(rep_held)
    val = held_texts + rep_held[:k_rep]; rng2.shuffle(val)
    assert val, "empty val shard: enable a corpus source or replay.measure_forgetting"
    vpath = train_dir / f"shard_{len(recs):05d}.parquet"
    pq.write_table(pa.table({"text": val}), vpath, row_group_size=16)
    manifest["val_shard"] = {"file": vpath.name, "docs": len(val), "note": "held out; nanochat's val split"}
    shares = [x["replay_share"] for x in recs[:-1]] or [x["replay_share"] for x in recs]
    manifest["totals"] = {"corpus_tokens": corpus_tokens, "replay_tokens": replay_tokens, "train_tokens": total,
                          "replay_share": replay_tokens / total, "replay_share_per_shard_min": min(shares),
                          "replay_share_per_shard_max": max(shares), "train_shards": len(recs)}
    for d in sorted(out.rglob("*.parquet")):
        h = hashlib.sha256()
        with open(d, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        manifest.setdefault("sha256", {})[str(d.relative_to(out))] = h.hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"mix: wrote {len(recs)} training shards + val to {train_dir}; {total:,} tokens, replay share "
        f"{manifest['totals']['replay_share']:.3f} (per shard {min(shares):.3f}-{max(shares):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys (dotted, YAML values)")
    ap.add_argument("--stage", default="all", choices=["all", "corpus", "mix"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test: at most this many documents per source")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    work = HERE / cfg["work_dir"]
    if a.limit:
        work = work / f"limit{a.limit}"
    if a.stage in ("all", "corpus"):
        counter = Counter(HERE / cfg["tokenizer"])
        for s in enabled_sources(cfg):
            build_corpus_source(s, cfg, counter, work, a.limit)
    if a.stage in ("all", "mix"):
        mix(cfg, work, a.limit)


if __name__ == "__main__":
    main()
