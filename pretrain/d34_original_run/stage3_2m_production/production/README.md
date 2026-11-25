# Production continuation after the four-arm decision

> In this recipe the coordinator steps below are scripted: `../prepare_corpus.sbatch` (full-corpus manifest and
> receipt), `../prepare_release.sbatch` (code manifest, winner manifest, static assets, release JSON) and
> `../launch_production.sh` (placement file, first leg); every `D34_*` variable comes from `config.sh`. This file is
> the original design note. The "four-arm decision" was a screen of batch 1M/2M x Muon LR 0.01414/0.02 from step
> 136,316; arm 3 (2M, 0.01414) won and is the only arm in this recipe.

These are separate production files; the ablation trainer and its snapshots
are unchanged. No job is submitted by creating or testing these files.
The coordinator selects a winner only after all four arms are complete and
comparable. This directory deliberately contains no automatic winner or tie
fallback policy.

## Clock and optimizer contract

Every winning terminal checkpoint represents canonical 1M step **138364** and
**94,685,102,080 total training tokens**, although its actual optimizer step
is 138364 for a 1M batch and 137340 for a 2M batch. The original horizon remains
canonical step **321817**, or **287,049,515,008 total training tokens**. Production
therefore trains another **192,364,412,928 tokens**.

| Winning batch | Additional optimizer updates | Final actual step |
|---|---:|---:|
| 1,048,576 | 183,453 | 321817 |
| 2,097,152 | 91,726 full updates plus one final 1M update | 229067 |

The final half-size update for a 2M winner uses eight accumulation microsteps
instead of sixteen. This is necessary to finish at the exact original token
horizon. All other updates keep the selected batch size. Device batch stays
eight, context 2048, and global world size eight, using 1×8 or 2×4 full Hopper
GPUs. Training stays FP8 tensorwise and validation BF16.

Every checkpoint restores all saved optimizer buffers and initial rates. The
optional `--production-matrix-lr` argument is an assertion of the selected
rate, never an override. Unlike the short ablation, production applies the
original LR multiplier to **all optimizer groups**, including non-Muon groups.
The multiplier remains 1 through canonical step 143917 and then decays toward
0.05. Muon momentum follows the original 0.97→0.90 schedule, and Muon weight
decay uses the same 1M reference amplitude and original cosine schedule.
Changing batch size on a subsequent production leg is rejected.

## Checkpoints and resumes

`production_runtime.py` stores actual step, canonical step, true cumulative
tokens, selected batch/rate, world size, and all eight rank-specific loader
cursors in `meta_*.json`. Subsequent legs read their own rank's cursor.
The existing loader still does not serialize its packing buffer or prefetched
batch: job-boundary resumes remain **approximate row-group resumes**, rather
than exact uninterrupted data-stream replay. The first production restart reads
its own rank's `prefetched_loader_state` from the selected original
`COMPLETE.json`. The checkpoint metadata stores only rank zero's cursor; it
must never be used as a substitute for the other seven ranks. The original
completion bytes are bound to the winner manifest and release by SHA256.
Missing, malformed, changed, or mismatched evidence fails before data loading.
Subsequent production checkpoints supply their own saved rank cursors as before.

Checkpoint files are written into a temporary directory. After all eight
optimizer shards exist, the model and optimizer files are moved to the output
directory, then metadata is published last. `checkpoint_state.py` selects only
complete production sets. The most recent three complete production checkpoints
are retained; deletion is restricted to files explicitly marked as production
checkpoints inside this new output directory. The selected ablation checkpoint
is kept in a separate source directory and is never pruned.

Deadline handling runs before validation and saves a full checkpoint. The
default deadline margin is 30 minutes. A final checkpoint saved immediately
before a deadline can be resumed just to perform final validation and publish
`BASE_DONE`; it does not perform an extra training update. A clean deadline
with no newly committed progress counts toward the failure limit. The coordinator
should ensure the chosen wall time leaves enough room for startup, a useful
training interval, validation, and checkpoint writing.

## Coordinator integration

Use `production_leg.sbatch` only after creating a trusted shell configuration
file and exporting its absolute path as `D34_PROD_CONFIG`. The wrapper sources
the file with automatic export so the same settings reach each node. Required
configuration variables are:

- `D34_PROD_DIR`: new shared production state/log directory.
- `D34_PRODUCTION_CODE`: deployed absolute directory containing these files.
- `D34_EXP_DIR`: frozen experiment root containing `upstream_snapshot/`.
- `D34_BASE_DIR`: shared base containing tokenizer and `base_checkpoints/`.
- `D34_PROD_TAG`: a new tag beginning with `d34prod`.
- `D34_PYTHON`, `D34_HF_HOME`: the frozen runtime and offline kernel cache.
- `D34_ENV_SETUP`: optional trusted runtime/module environment setup file.
- `D34_WINNER_CHECKPOINT_DIR`, `D34_WINNER_STEP`, `D34_WINNER_BATCH`,
  `D34_WINNER_LR`: selected terminal checkpoint and its exact settings.
- `D34_WINNER_COMPLETION`, `D34_WINNER_COMPLETION_SHA256`: copied original
  selected `COMPLETE.json` and its exact byte SHA256, matching the release.
- `D34_GPUS_PER_NODE`: 4 on two nodes, or 8 on one node.
- `D34_SHARD_MANIFEST`, `D34_DATA_ROOT`: indexed corpus manifest and raw data.
- `D34_PROD_RELEASE`: coordinator's decision JSON, described below.
- `D34_FLEET_DIR`, `D34_CHECKPOINT_DIR`, `D34_RUNTIME_ROOT`,
  `D34_ASSET_MANIFEST`, `D34_ASSET_RECEIPT`: the existing static ablation asset
  environment. Keep `D34_CHECKPOINT_DIR` pointing to protected step 136316;
  the selected winner has its separate variable above.
- `D34_FULL_CORPUS_RECEIPT`: destination receipt from a completed allocated
  SHA256 verification of every full-corpus shard.
- `D34_PROD_SBATCH_ARGS_FILE`: one approved Slurm placement argument per line,
  including account, nodes, GPU count, CPU/memory, wall time and output path.
  Empty lines and lines starting with `#` are ignored. No shell evaluation is
  performed on these arguments.

`D34_PROD_NO_CHAIN=1` disables automatic next-leg submission. Otherwise the
wrapper queues exactly one successor with `afterany` on itself. A production
directory lock prevents simultaneous writers. Each successor checks `BASE_DONE`
and `CHAIN_STOP`; three consecutive failures stop the chain. A missing EXIT
record from a killed predecessor also counts as a failure. Only the precise
successor job ID created by the current leg can be cancelled on completion or
chain stop. No user-wide job operation is used.

The release JSON must contain:

```json
{
  "status": "approved_for_production",
  "validation_status": "all_four_arms_complete_and_comparable",
  "actual_checkpoint_step": 138364,
  "total_batch_size": 1048576,
  "actual_matrix_lr": 0.01414213562373095,
  "winner_checkpoint_manifest": "/absolute/path/to/winner_manifest.json",
  "winner_checkpoint_manifest_sha256": "REPLACE_WITH_ACTUAL_SHA256",
  "winner_completion": "/absolute/path/to/selected/COMPLETE.json",
  "winner_completion_sha256": "REPLACE_WITH_ACTUAL_SHA256",
  "production_code_manifest_sha256": "REPLACE_WITH_ACTUAL_SHA256",
  "full_corpus_manifest": "/absolute/path/to/full_corpus_manifest.json",
  "full_corpus_manifest_sha256": "REPLACE_WITH_ACTUAL_SHA256"
}
```

Those numerical values illustrate the low-LR 1M arm; the coordinator must write
the actual selected winner. The wrapper checks them against its configuration
and the selected terminal checkpoint metadata. This release file is an
explicit coordinator gate, not an independent reanalysis of the four arms.

Allocated preflight is implemented. `preflight.sh` first verifies the existing
static ablation assets, then `production_preflight.py` verifies the separately
frozen production code, the selected winner's complete ten-file SHA256 manifest,
the selected original completion record, the current production resume metadata
and rank cursors, and the complete corpus receipt. The initial source directory
and actual step must be the verified winner. Both preflight and every first-leg
trainer rank re-read the release-pinned winner manifest and metadata SHA256,
then validate the same completion mapping. Checks cover the source step/tag,
terminal step, batch-derived update count, exact 2 Gi-token budget, canonical
clock, actual Muon LR, all ten checkpoint filenames, and exactly eight rank
records with the expected consumed counts, hashes, and valid rank-specific
cursors. The chain release check also validates this binding before queuing a
successor. The selected checkpoint is not passed through the ablation validator's
hardcoded step-136316 contract. In a separate subsequent Slurm step, the fleet's
existing eight-rank collective probe checks Hopper hardware, Python/Torch/CUDA
versions, BF16 compute, offline FA3, manifest consistency and NCCL all-reduce.
Training uses another port and starts only after both phases pass.

Before approval, run `freeze_code.py` locally to create `code_manifest.json` and
bind its SHA256 in the later release. Run `winner_manifest.py create` in a CPU
allocation on the selected checkpoint's source cluster, passing the original
completion record explicitly:

```
python winner_manifest.py create --checkpoint-dir /absolute/selected/checkpoint --step SELECTED_ACTUAL_STEP --completion /absolute/selected/branch/COMPLETE.json --output /absolute/winner_manifest.json
```

Copy that v2 manifest, the unchanged `COMPLETE.json`, and all ten checkpoint
files. `winner_manifest.py verify` also requires `--completion`. Production
preflight verifies destination checkpoint bytes and caches a filesystem
fingerprint receipt; it always rechecks completion bytes, even on a cache hit.
Old v1 winner manifests without completion evidence are rejected and must be
regenerated in an allocation. No winner manifest or release is preselected by
these files.

Full corpus identity is required separately from the static ablation manifest,
which covers only 129 parquet files. In a source CPU allocation, run:

```
python corpus_manifest.py create --data-dir "$D34_DATA_ROOT" --output /absolute/path/to/full_corpus_manifest.json --workers 4
```

If production runs on another cluster, copy that manifest there. In a destination CPU allocation, run:

```
python corpus_manifest.py verify --data-dir /absolute/path/to/destination/base_data_climbmix --manifest /absolute/path/to/full_corpus_manifest.json --receipt /absolute/path/to/full_corpus_receipt.json --workers 4
```

Both commands stream with at most four 8 MiB reads at once; four CPUs and 8 GiB
RAM are sufficient. They keep a per-file partial ledger and reuse unchanged
file hashes after interruption. Source creation binds all 6,543 canonical
basenames, sizes and SHA256 hashes without claiming an unverified upstream
revision. Destination verification must complete before launch. Production
preflight requires the receipt to match the released full manifest and every
file's current filesystem fingerprint; it fails on missing, changed or partially
staged data instead of rehashing 558 GiB while holding eight GPUs.

The original production ran on one 8×H100 node; 2×4 H100 placements are also
accepted. The ablation's initial 1945–2072 staging window
is sufficient for the short ablation but **insufficient for full production**.
This implementation requires the full verified corpus for production. Sparse
window mode is deliberately rejected until a separate measured coverage bound
and runtime stop guard are provided. An unstaged shard never changes the data order.

CORE and sampling are disabled by default in the leg wrapper; validation runs
every 250 canonical 1M steps and saving every 1200 canonical steps. Set
`D34_PROD_CORE_EVERY` to enable a specific CORE cadence, accounting for its
runtime in the deadline margin. `BASE_DONE` ends this chain; downstream
finetuning/evaluation remains coordinator-owned.

## Validation status

Sixteen local tests pass. The existing clock, scheduler, checkpoint publication,
chain-failure and SHA/corpus receipt checks remain covered. New regressions
exercise distinct first-resume rank cursors for both batch sizes, the actual
completed arm 2/4 records, missing and altered completion bytes after a cached
receipt, malformed or mismatched protocol/rank evidence, release/manifest/source
and metadata bindings, and unchanged subsequent-production cursor selection.
All production Python files and shell scripts pass syntax checks.

These are control-flow/filesystem/numerical-schedule tests, not GPU model
training. An allocated production smoke still needs to verify actual winner
loading, scheduled rates, checkpoint publication and a subsequent real resume.
