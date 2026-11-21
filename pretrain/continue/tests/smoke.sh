#!/usr/bin/env bash
# CPU smoke test of the continued-pretraining trainer and of the hand-off to the reranker (a few minutes, no GPU).
#   0. the Gaggle tokenizer (four small files from the Hub) and a synthetic corpus: training shards + an eval set;
#   1. a 2-layer model pre-trained for 3 steps, laid out like a released checkpoint dir (with its tokenizer);
#   2. scripts/base_train.py --config tests/smoke.yaml: an unknown key is refused; the file's values are used; a
#      command-line flag overrides the file; the run starts from the checkpoint, with its tokenizer, which is copied
#      into the run's dir; the meta leaves out ve_every when it is 2;
#   3. scripts/base_eval.py --checkpoint-dir: bpb of the starting checkpoint and of the run (newest and a given step);
#   4. the run's dir as a reranker backbone, with the paper's pinned nanochat: the loader takes the newest step or
#      `model_step`, the tokenizer loads from the dir, and eval/rerank_eval.py scores with a reranker checkpoint
#      whose paths are relative to reranker/, as training writes them.
# Needs: bash setup.sh; a Python with nanochat's dependencies, pyyaml, huggingface_hub and transformers (PYTHON=...).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
W=work/smoke
REPO=$(cd ../.. && pwd)
export CUDA_VISIBLE_DEVICES="" NANOCHAT_BASE_DIR=$W/base PYTHONPATH="third_party/nanochat${PYTHONPATH:+:$PYTHONPATH}"
rm -rf "${W:?}"; mkdir -p $W/base
LOG=$W/train.log
fail() { echo "smoke: FAILED: $*"; exit 1; }

# 0. tokenizer and data
$PY - <<'PYEOF' || fail "tokenizer download / synthetic data"
import os, random, shutil
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
tok = "work/smoke/tok"
for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt"):
    hf_hub_download("castorini/gaggle-nanochat-pretrained-climbmix-20260924", f, local_dir=tok)
rng = random.Random(0)
words = ("the of and to in is was for on that with by as at from it an be this are or which one all were their "
         "search model data query document passage retrieval ranking relevance corpus training language").split()
doc = lambda: " ".join(rng.choice(words) for _ in range(rng.randint(40, 300))).capitalize() + "."
def shards(d, n, docs):
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        pq.write_table(pa.table({"text": [doc() for _ in range(docs)]}), f"{d}/shard_{i:05d}.parquet", row_group_size=16)
shards("work/smoke/data/train", 3, 200)     # the last shard is nanochat's val split
shards("work/smoke/data/eval", 2, 40)       # base_eval --eval bpb scores both shards
PYEOF

# 1. the starting checkpoint, laid out like a released one
$PY -m scripts.base_train --depth 2 --aspect-ratio 64 --head-dim 64 --window-pattern SL --max-seq-len 256 \
  --device-batch-size 2 --total-batch-size 1024 --num-iterations 3 --eval-every -1 --eval-tokens 1024 \
  --core-metric-every -1 --sample-every -1 --model-tag smoke_init --data-dir $W/data/train \
  --tokenizer-dir $W/tok/tokenizer > $LOG 2>&1 || fail "pre-training the tiny model (see $LOG)"
INIT=$W/base/base_checkpoints/smoke_init
cp -r $W/tok/tokenizer $W/tok/tokenizer.json $W/tok/tokenizer_config.json $INIT/

# 2. continued pretraining from a config file
printf 'num-iterations: 5\nnot-a-flag: 1\n' > $W/bad.yaml
if $PY -m scripts.base_train --config $W/bad.yaml >> $LOG 2>&1; then fail "an unknown config key was accepted"; fi
grep -q "not flags of this script: \['not_a_flag'\]" $LOG || fail "no error message for the unknown key"
$PY -m scripts.base_train --config tests/smoke.yaml --num-iterations 10 >> $LOG 2>&1 || fail "the continued run (see $LOG)"
grep -q "(continue) initialised from $INIT step 3" $LOG || fail "did not start from the checkpoint"
RUN=$W/base/base_checkpoints/smoke_cpt
[ -f $RUN/model_000010.pt ] && [ -f $RUN/model_000005.pt ] && [ ! -f $RUN/model_000020.pt ] \
  || fail "--num-iterations on the command line did not override the file"
[ -f $RUN/tokenizer.json ] && [ -f $RUN/tokenizer_config.json ] && [ -f $RUN/tokenizer/tokenizer.pkl ] \
  || fail "the tokenizer was not copied into the run's dir"
$PY - <<'PYEOF' || fail "the run's meta"
import json
m = json.load(open("work/smoke/base/base_checkpoints/smoke_cpt/meta_000010.json"))
uc = m["user_config"]
assert uc["matrix_lr"] == 0.002 and uc["weight_decay"] == 0.021764 and uc["literal_hparams"] is True, uc
assert uc["num_iterations"] == 10 and uc["data_dir"] == "work/smoke/data/train", uc
assert uc["tokenizer_dir"] == "work/smoke/base/base_checkpoints/smoke_init/tokenizer", uc
assert "ve_every" not in m["model_config"], m["model_config"]
PYEOF

# 3. base_eval on checkpoint dirs (each with its own tokenizer; $NANOCHAT_BASE_DIR has none)
for args in "$INIT" "$RUN" "$RUN --step 5"; do
  NANOCHAT_DATA_DIR=$W/data/eval $PY -m scripts.base_eval --eval bpb --checkpoint-dir $args \
    --device-batch-size 2 --split-tokens 4096 > $W/eval.log 2>&1 || fail "base_eval --checkpoint-dir $args (see $W/eval.log)"
  grep -q "^val bpb: [0-9.]*$" $W/eval.log || fail "no bpb for $args"
  echo "base_eval --checkpoint-dir $args: $(grep -E '^(train|val) bpb' $W/eval.log | tr '\n' ' ')"
done

# 4. the run's dir as a reranker backbone (fresh process: the loader imports nanochat from nanochat_src)
cd "$REPO"
PYTHONPATH= $PY - <<'PYEOF' || fail "reranker hand-off"
import os, sys, torch
sys.path.insert(0, "reranker/src"); sys.path.insert(0, "eval")
from transformers import AutoTokenizer
from modeling_nanochat_upstream import UpstreamNanochatReranker
from prompt_builder_nc import setup_tokenizer, yes_no_ids
import rerank_eval
run = "pretrain/continue/work/smoke/base/base_checkpoints/smoke_cpt"
src = os.path.abspath("reranker/third_party/nanochat-92d63d4")          # the paper's pinned nanochat
tok = setup_tokenizer(AutoTokenizer.from_pretrained(run))
yes, no = yes_no_ids(tok, "nochat", None)
newest = UpstreamNanochatReranker(run, yes_id=yes, no_id=no, dtype=torch.float32, nanochat_src=src)
pinned = UpstreamNanochatReranker(run, yes_id=yes, no_id=no, dtype=torch.float32, nanochat_src=src, step=5)
assert (newest.step, pinned.step) == (10, 5), (newest.step, pinned.step)
# a reranker checkpoint as reranker/src/train_nc.py writes it: cfg paths relative to reranker/
cfg = {"model_name": "../" + run, "nanochat_src": "third_party/nanochat-92d63d4", "model_step": 5,
       "backbone_impl": "upstream", "prompt": "nochat", "max_doc_len": 64, "max_query_len": 16}
score = rerank_eval.load_ours_pointwise(None, torch.device("cpu"), torch.float32,
                                        {"cfg": cfg, "model": pinned.state_dict()})
s = score("what is a corpus", ["a corpus is a collection of documents", "the model was trained"], 2, 128)
assert len(s) == 2 and all(map(lambda x: x == x, s)), s
print("reranker: newest step", newest.step, "| model_step", pinned.step, "| rerank_eval scores", [round(x, 4) for x in s])
PYEOF
echo "smoke: OK"
