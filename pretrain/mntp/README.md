# Bidirectional (MNTP) variant of the d34 backbone

The recipe behind [`castorini/gaggle-nanochat-pretrained-climbmix-bidirectional-20260929`](https://huggingface.co/castorini/gaggle-nanochat-pretrained-climbmix-bidirectional-20260929)
(paper Sec. 4, "Bidirectional variant"; scored in the pre-training comparison and fine-tuned in Sec. 6.3).
Code by Lingwei Gu.

It continues the causal run from step 209,158 (245.3B tokens) to the original one-pass horizon
(step 229,067, 287.05B tokens): 41.75B tokens, 19,909 updates of 2,097,152 tokens. The data stream continues from
the checkpoint's eight per-rank cursors, so it sees exactly the tokens the causal run consumed after 209,158, and the
final causal checkpoint is a matched control.

## What changes from the causal trainer

Everything else is identical: model, FP8, batch, learning rates, schedule.

- **Attention** is bidirectional (`NANOCHAT_BIDIRECTIONAL=1`): sliding windows `(w, 0)` become `(w, w)`. The model
  is still a decoder; `upstream_snapshot_bidir/nanochat/{gpt,flash_attention}.py` carry the change.
- **Objective**: LLM2Vec-style masked next-token prediction. 20% of input positions (never BOS / position 0) are
  replaced by `<|output_end|>` (id 32767, unused in pre-training). The output at position t predicts the original
  token at t+1, only where t+1 was masked. There is no 80/10/10 keep, because it would leak under bidirectional
  attention.
- **Optimizer**: fresh (no optimizer state was saved at 209,158), at the run's fixed learning rates (Muon 0.0141).
  A 200-step linear re-warmup reaches 0.26 of peak, where the causal schedule stood; then the original linear decay
  to 0.05 of peak.
- **Evaluation**: held-out MNTP masked-token loss and accuracy (fixed masks) every 250 canonical steps. CORE, bpb and
  sampling are causal metrics, so they are off.
- **Start-up check**: FA3 non-causal windows must match the SDPA band reference, or the run stops.

## Run

You need the causal checkpoint at step 209,158 (canonical step 282,000, 245.3B tokens). It is not released and will
not be, so it has to come from your own run of the causal recipe as it ran, stage by stage (`../d34_original_run/README.md`;
the MNTP trainer resumes a stage-3 production checkpoint and requires its metadata):
- Stage 3 saves every 1,200 canonical steps and keeps only the newest three (`--production-keep-latest=3`). 282,000
  is a save point.
- When it is written, copy `model_209158.pt` and `meta_209158.json` out of `base_checkpoints/<tag>/`. The meta file
  carries the eight per-rank data cursors the continuation resumes from. Alternatively, raise
  `--production-keep-latest` in `d34_original_run/stage3_2m_production/production/node_launch.sh` before stage 3 starts.

Then, on one node with 8 H100/H200 GPUs:

```bash
SRC_DIR=<dir with model_209158.pt and meta_209158.json> NANOCHAT_BASE_DIR=<the causal recipe's base dir> \
OUT_DIR=<logs dir> bash pretrain/mntp/launch_mntp.sh
```

About 340 H100-hours. Re-running the same command resumes from the newest checkpoint.

`smoke/smoke_bidir.py` checks the attention patch and the masking before a long run. SDPA bands must equal a dense
reference, FA3 must equal SDPA, a future token must change earlier logits, and masks must never expose a target.
A 60-step run at production learning rates should take held-out MNTP loss from about 4.95 to about 2.36.

## Layout

| Path | Contents |
|---|---|
| `trainer/` | the production trainer with the MNTP objective (`bidir_mntp.py`), clock, checkpoints and chain helpers |
| `upstream_snapshot_bidir/` | nanochat `92d63d4` with bidirectional sliding-window attention (MIT license) |
| `smoke/smoke_bidir.py` | the pre-flight equivalence and masking checks |
| `launch_mntp.sh` | single-node launcher with the run's exact arguments |
