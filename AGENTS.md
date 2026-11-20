# Rules for coding agents

This repository is Project Greenhouse's home. Everything in it so far reproduces its first paper (arXiv:2610.11922).
New work goes in alongside it, with its own configs and records; nothing may change what the paper's rows reproduce.

## Do not change the paper's records

- Configs under `reranker/configs/{sft,backbones,data,attention,curation,loss}/` are the paper's rows. Do not edit
  them. New experiments go in `reranker/configs/experiments/`.
- `paper/manifest.yaml`, `paper/results/recorded.json`, `reranker/data_prep/SHA256SUMS`, `eval/candidates/DIGESTS`
  and `eval/candidates/runs/` are records of the original runs. Do not regenerate them to make a check pass.
- `reranker/third_party/nanochat-92d63d4/`, `pretrain/nanochat/` and `pretrain/mntp/upstream_snapshot_bidir/` are
  pinned upstream snapshots. Do not edit them.

## Benchmaxxing

`benchmaxxing/` holds experiments that train on the evaluation corpora (never on their queries or judgements). Follow
the rules in `benchmaxxing/README.md`. In short: label such models corpus-aware, not zero-shot; pick no checkpoint
by test score; keep their results out of `paper/`. The CPT trainer is built from upstream nanochat by
`benchmaxxing/cpt/setup.sh`; change it through `benchmaxxing/cpt/nanochat_cpt.patch`, not in `third_party/`.

## Never commit

- **Document text from Robust04 or TREC-News.** That means the candidate files (`retrieve_results_*_top100.jsonl`),
  listwise request files, model inputs or outputs that quote documents. They are licensed by NIST and may not be
  redistributed. Doc IDs, qrels, run files and scores are fine.
- **Large artifacts:** anything under `reranker/{models,data,checkpoints}/`, `data/`, `eval/{qrels,runs,results}/`,
  `exports/` or `benchmaxxing/cpt/{third_party,models,work,data,results}/`.
- **Credentials:** tokens, keys, `.env` files.

## How to work

- Fine-tuning runs from `reranker/` with `bash launch/train.sh <config> <n_gpus>`. Evaluation runs from the repo root
  with `bash eval/run_eval.sh <scorer> <model> <tag>`. The scorer for each row is in `paper/manifest.yaml`.
- A fine-tuning checkpoint is ~36 GB with optimizer state. Strip finished runs with
  `python reranker/tools/strip_optimizer.py` (12 GB, still evaluable, no longer resumable). Keep at most two
  in-flight checkpoints per run (`keep_ckpts: 2`).
- Judge differences against the measured run-to-run noise: about 0.001 on benchmark means and 0.003 on single
  collections. Do not report smaller differences as effects.
- Do not evaluate intermediate checkpoints on the test collections to pick a checkpoint. Every row uses its final
  checkpoint.
- Check a reproduced row with `python paper/check_results.py --results eval/results --tag <row>=<tag>`.

## Skills

`.claude/skills/` holds the procedures:
- `reproduce-paper-row`
- `evaluate-reranker`
- `prepare-data`
- `pretrain-backbone`
- `model-soup`

Load the matching one before starting any of those tasks.
