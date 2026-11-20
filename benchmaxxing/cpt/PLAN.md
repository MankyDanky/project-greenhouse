# Iteration 1: continued pretraining on the evaluation corpora

**Status (2026-11-20):** the plan, data preparation, training, evaluation and export scripts are ready. Data
preparation and the training pipeline pass CPU smoke tests (`tests/`). No GPU run has been made yet. The 0.6B
Gaggle model, on which the sweeps run, is not released yet: its configs carry placeholders (repository, step,
original weight decay) that `train.py` refuses to run with. The 3B runs can start now.

**Question.** If a Gaggle base model keeps pre-training on the raw documents of the collections the reranker is
evaluated on, does the same reranker fine-tuning then score higher? If so, how much of the gain comes from those exact
documents, and how much from simply training longer?

**Labelling.** These models see the evaluation corpora (never the queries or judgements). Their BEIR results are
**corpus-aware, not zero-shot**, and must be labelled that way wherever they are reported. TREC DL is in-domain for
every Gaggle reranker anyway, since its fine-tuning data (RLHN-250K) is built on MS MARCO. With `exclude_ood: true`
the four collections the paper counts as out of domain (TREC-COVID, NFCorpus, Robust04, TREC-News) stay out of the
mix, so their results are not corpus-aware.

---

## 1. Decisions

| # | Question | Iteration 1 | Section |
|---|---|---|---|
| 1 | When to mix the corpus in | **After pre-training (CPT)**, from the released Gaggle base models: they are the only checkpoints available. Within the run, corpus and replay are interleaved from start to end. | 2 |
| 2 | What to mix in | **Raw document text** of all the evaluation corpora (all seven BEIR collections, Robust04 and TREC-News included; MS MARCO v1 and v2), **balanced** between BEIR, MS MARCO v1 and MS MARCO v2 (1 : 1 : 1). No queries, no judgements, no prompt/answer format. An option (`exclude_ood`) keeps the four out-of-domain collections out. | 3 |
| 3 | Replay | **25% of training tokens**: pre-training data (ClimbMix), interleaved at a constant ratio, added on top of the corpus. A 0 / 5 / 25 / 50% sweep on the 0.6B model checks it against a rule fixed in advance. | 4 |
| 4 | Models | **Gaggle 3B** (`castorini/gaggle-nanochat-pretrained-climbmix-20260924`) and **Gaggle 0.6B** (to be released). Sweeps run on the 0.6B model. | 6 |
| 5 | Learning rate | Peak **0.1 × the original run's peak**, 1% warmup, linear decay to 0 over the second half. Probed at 0.03 / 0.1 / 0.3 on the 0.6B model. | 5 |
| 6 | Reranker stage | Unchanged: the paper's run-1 recipe for the 3B model; the same recipe with a memory-sized micro-batch for the 0.6B model. Only the starting weights change. | 6 |

Everything above is a YAML setting: the mix in `configs/data/`, the training run in `configs/train/`.

---

## 2. When to mix

**After pre-training.** The released Gaggle checkpoints are the ends of their runs (the 3B model finished at about
0.07 × its peak learning rate), and no earlier checkpoint is released. So the corpus enters after pre-training:
continued pretraining (CPT) from the released weights, with its own short warmup and decay.
- **It is affordable.** One arm of the main mix is about 4.4B tokens, about 1.5% of the 287B of pre-training.
- **It isolates the question.** The base model and the reranker recipe stay fixed; only the CPT stage is added.

**Within the CPT run**, corpus and replay documents are interleaved from the first step to the last. How the replay
share moves over the run is a setting: `replay.schedule: constant | front_loaded | back_loaded` with `replay.skew`.
The default is constant (section 4). A back-loaded schedule, with more replay as the learning rate decays, is left
as an ablation.

---

## 3. What to mix

**Raw documents of the evaluation corpora, and nothing else.** Every document is used as plain text with the
language-modelling loss on every token:
- **Format:** `title\n\ntext`, or the text alone when there is no title. MS MARCO passages have no title.
- **Length:** documents over 3,500 characters (about 800 tokens) are split at a paragraph, then sentence, then word
  boundary.
- **Exact duplicates** are dropped.
- **Never used:** queries, relevance judgements, BM25 candidate lists, or any selection by query.

The 12 test sets read 9 corpora, in three groups:

| Group | Corpus | Test sets | Tokens (one pass) | Most passes |
|---|---|---|---:|---:|
| BEIR | TREC-COVID, NFCorpus, FiQA, SciDocs, SciFact | the same names | ~60M together | 4 |
| BEIR | Robust04, TREC-News | the same names | ~851M together | 1 |
| MS MARCO v1 | MS MARCO v1 passages | DL19, DL20 | ~715M | 2 |
| MS MARCO v2 | MS MARCO v2 passages | DL21, DL22, DL23 | ~8.2B | 1 |

**Balance.** The groups enter at a ratio set in the config (`groups:`), 1 : 1 : 1 by default. The group that runs out
first, at its sources' most passes, sets the corpus size; every other group contributes its ratio of that. Below one
pass a source contributes a fixed random subset of its documents (by hash, so a smaller amount is always a subset of
a larger one). With the defaults:

| Group | In the mix | Of what is available |
|---|---:|---|
| BEIR | ~1.09B | everything: the open corpora 4 times, Robust04 and TREC-News once (BEIR sets the size) |
| MS MARCO v1 | ~1.09B | about 1.5 passes: every passage once, half of them twice |
| MS MARCO v2 | ~1.09B | about 13% of the passages |
| **Corpus** | **~3.3B** | with 25% replay: **~4.4B tokens**, about 4,000 optimizer steps of 2^20 tokens |

Notes:
- **Why balance.** Taken whole, MS MARCO v2 alone would be about 90% of the corpus and BEIR about 2%.
  `configs/data/ir_all_r25.yaml` (`ratio: all` for every group) is that option; no run uses it.
- **MS MARCO v2 exposure is partial.** About 13% of its passages, and so about 13% of the DL21–23 candidates, are
  in the mix: a fixed random subset chosen by hash, independently of any query. A larger v2 ratio raises it.
- **The small BEIR corpora get 4 passes.** At one pass NFCorpus would be a negligible share. Muennighoff et al. (2023)
  find that up to about 4 passes over repeated data are nearly as useful as fresh data. The passes follow one
  another, each spread evenly over its own part of the run and in its own order.
- **Robust04 and TREC-News are always included.** They are read from Pyserini's prebuilt BEIR indexes, the same ones
  the evaluation uses (needs `pyserini` and Java 21). Their text is licensed by NIST and may not be redistributed:
  never publish a built mix, and check the licence before releasing a model trained on one.
- **Option: keep the out-of-domain collections out.** The paper counts four BEIR collections as out of domain for
  RLHN-250K: TREC-COVID, NFCorpus, Robust04 and TREC-News (marked `ood: true`). `exclude_ood: true` leaves them out
  of the mix entirely, so they stay out of domain for the CPT model too (`configs/data/ir_r25_no_ood.yaml`, with
  train configs `gaggle3b_ir_r25_no_ood` and `gaggle06b_ir_r25_no_ood`; no run in section 6 uses it). The mix is
  then sized without them. In-domain BEIR (FiQA, SciDocs, SciFact) is only ~18M tokens, ~72M at 4 passes, so this
  option has its own ratios, 1 : 4 : 4: BEIR ~72M (it sets the size), MS MARCO v1 ~0.29B (~40% of its passages),
  MS MARCO v2 ~0.29B (~3.5%). Corpus ~0.65B tokens, ~0.86B with 25% replay, about 780 steps of 2^20 tokens.
- **Every knob is in the YAML:** each source's most passes (`passes`), the group ratios (`groups`), out-of-domain
  collections in or out (`exclude_ood`), sources to leave out after sizing (`exclude_sources`), and the replay
  (section 4).

**Later candidates, not in iteration 1:**
- the other BEIR corpora, to separate "retrieval domain" from "these exact documents";
- synthetic queries for the corpus documents (doc2query style), which turns CPT into something close to fine-tuning.

---

## 4. Replay: how much, how, and from where

**Answer: 25% of the CPT tokens are pre-training data, mixed in at a constant ratio, document by document, on top of
the corpus.**

### What the literature says

| Work | Setting | Finding on replay or learning rate |
|---|---|---|
| Ibrahim et al. 2024 ([2403.08763](https://arxiv.org/abs/2403.08763)) | CPT of 405M–10B models, LR re-warmed and re-decayed | **5%** replay enough for a weak shift (web → web); **25%** for a strong one (English → German). A higher re-warmed LR adapts more and forgets more. |
| Gupta et al. 2023 ([2308.04014](https://arxiv.org/abs/2308.04014)) | re-warming for CPT | Re-warming raises the loss on the old data at first; the peak LR trades adaptation against forgetting. |
| Guo et al. 2024 ([2406.14833](https://arxiv.org/abs/2406.14833)) | the "stability gap" in CPT | Performance first drops, then recovers. A mixture close to the pre-training data shrinks the drop. |
| Parmar et al. 2024 ([2407.07263](https://arxiv.org/abs/2407.07263)) | a recipe for continued pretraining | A general blend first, then a target-focused blend switched in late in the decay (at about ηmax/5). |
| Bethune et al. 2025 ([2502.06042](https://arxiv.org/abs/2502.06042)) | scaling laws for forgetting with pre-training data injected | Even **1%** of pre-training data limits forgetting substantially; smaller models forget more. |
| EvoLM, 2025 ([2506.16029](https://arxiv.org/abs/2506.16029)) | CPT then SFT, measured after SFT | A moderate replay share (**16%** in their sweep) gave the best results after fine-tuning. |
| Abbes et al. 2025 ([2508.01908](https://arxiv.org/abs/2508.01908)) | replay vs. model size in CPT | Replay around **25%** is a compute-efficient way to limit forgetting. |
| D-CPT Law ([2406.01375](https://arxiv.org/abs/2406.01375)); CMR ([2407.17467](https://arxiv.org/abs/2407.17467)) | scaling laws for the domain/general ratio | The best ratio can be predicted from small runs: a reason to sweep on the small model. |
| Domain CPT practice | Code Llama, Llemma, DeepSeekMath, PMC-LLaMA, Meditron | General text at about **1–10%** of the domain mix (Code Llama 7%, DeepSeekMath 10% Common Crawl, Meditron about 1%). |
| Corpus-aware retrieval pretraining | coCondenser, COCO-DR, GPL, Llama2Vec | Unsupervised training on the target corpus improves retrieval on that corpus: the precedent for this project. |

None of these works uses Muon, which trains the Gaggle models' matrix weights. That is one reason for the LR probe
(section 5).

### Why 25%, and why constant

- **Our shift sits between the two cases of Ibrahim et al.** It stays in English but is narrower text: web passages,
  newswire, biomedical, scientific and financial documents. That puts it between their 5% and 25%. We take the safe
  end because the reranker needs general knowledge (BEIR rewards it), so forgetting costs the final metric, not just
  a language-modelling benchmark.
- **Ours is true replay.** The replay comes from ClimbMix shards the 3B model was trained on (shards 0–5899; its
  data loaders were past shard 5,949 at release). Per token, that protects better than other general text, so the
  sweep may well find that 5% is enough. `train.py --check` reads where a starting model's data loader stopped and
  warns if part of the replay range is text it never saw (check this for the 0.6B model on release).
- **The cost is modest.** 25% replay means a third more tokens than none.
- **Constant ratio, interleaved document by document, fixed when the data is built.** Every stretch of the run sees
  the same share, so a run stopped early or preempted is still the same experiment. The trainer only reads shards in
  order; `manifest.json` records the realised share of every shard.
- **Added on top, not substituted.** Every replay arm trains on the same corpus documents. Arms with less replay use
  a prefix of the same replay stream, so the only thing that changes between arms is the amount of replay.

### Checking it: the replay sweep (0.6B model)

Arms `ir_r00`, `ir_r05`, `ir_r25`, `ir_r50`: same corpus, same learning rate, 0 / 5 / 25 / 50% replay. Measured
on held-out text in bits per byte (bpb), with `evaluate.py`:
- **gain:** the drop in held-out corpus bpb;
- **forgetting:** the rise in held-out ClimbMix bpb (pre-training validation shard 6542, never trained on);
- **CORE**, the general-ability benchmark the base models were selected by.

**Rule, fixed in advance:** use the smallest replay share that
1. removes at least half of the 0% arm's forgetting, **and**
2. keeps at least 90% of the 0% arm's held-out corpus gain.

If no share meets both, stay at 25%. Because small models forget more (Bethune et al.), the 0.6B result is, if
anything, conservative for the 3B model.

---

## 5. Optimisation

| Setting | Value | Why |
|---|---|---|
| Peak LR | 0.1 × the original run's peak, for all four parameter groups (Muon matrices, embeddings, unembedding, scalars) | The 3B model ended at about 0.07 × peak, so this is a mild re-warm: enough to adapt, little to forget (Ibrahim et al.; Gupta et al.). |
| LR probe | 0.03 / 0.1 / 0.3 × on the 0.6B model at 25% replay | Muon is untested in the CPT literature. 0.03 is below where pre-training ended; 0.3 is a real re-warm. |
| Schedule | 1% linear warmup, constant, linear decay to 0 over the second half | Pre-training decayed over its last 55%. Decaying to 0 gives a finished model, as in the replay papers. |
| Batch | 2^20 tokens per step (2,048-token sequences) | About 4,000 steps on the main mix. Pre-training ended at 2^21. |
| Weight decay | the original run's Muon amplitude, as is (3B 0.021764; the 0.6B value is filled in on release); cosine to 0 over the run | nanochat would otherwise rescale it for the short horizon (by tens of times). |
| Length | one pass: 95% of the mix's tokens | nanochat's loader would wrap into a second pass, and its packing crops a few percent of tokens. |
| Precision | FP8 matmuls, as in pre-training (`fp8: false` on GPUs older than H100) | |
| Optimizer state | fresh | The released checkpoints carry no optimizer state. |

`train.py` derives all of it from the checkpoint's `meta_<step>.json` and the YAML (`train.py --check` prints the
result). nanochat's own batch-size and horizon rescaling is switched off with `--literal-hparams`.

**LR rule, fixed in advance:** take the largest probe LR whose held-out ClimbMix bpb rises by at most 1% (relative)
and whose CORE drops by at most 0.01. If none qualifies, use 0.03. To change it, set `lr.scale` in
`configs/train/gaggle06b_ir_r25.yaml` (every 0.6B arm inherits it) and in `configs/train/gaggle3b_ir_r25.yaml`.

---

## 6. Runs

| Phase | Model | Arms (`configs/train/`) | Reranker fine-tune | Decides |
|---|---|---|---|---|
| 0 | – | `tests/smoke_data.yaml`, `tests/smoke_train.sh` (CPU) | – | the pipeline runs |
| 1 | 0.6B | `gaggle06b_ir_r25_lr003`, `gaggle06b_ir_r25`, `gaggle06b_ir_r25_lr03` | no | LR (rule in section 5) |
| 2 | 0.6B | `gaggle06b_ir_r00`, `_r05`, `_r25` (from phase 1), `_r50` | no | replay share (rule in section 4) |
| 3 | 0.6B | the chosen arm, `gaggle06b_control`, `gaggle06b_ir_r25_minus_scifact` | baseline (no CPT) + these three | does CPT help the reranker; how much is the corpus |
| 4 | 3B | `gaggle3b_ir_r25` (with the chosen share and LR), `gaggle3b_control` | these two; the baseline is the paper's run 1 | the main result |

The arms:
- **Control (replay only).** The same number of tokens and the same schedule, all of it ClimbMix. It separates the
  effect of the corpus from the effect of training (and re-annealing) longer. Set `replay.tokens` in
  `configs/data/control_replay_only.yaml` to the chosen arm's `totals.train_tokens` from its `manifest.json` before
  building it.
- **Leave-one-out (SciFact).** The full mix without SciFact's corpus. The mix is sized with SciFact counted, so every
  other source contributes exactly what it does in the full mix. The SciFact gain of the full arm over this one is
  what seeing SciFact's own documents adds.
- **Baselines.** For the 3B model, the paper's run 1 (`castorini/gaggle-reranker-run1-20260926`; DL 0.641, BEIR
  0.544). The reranker on the untouched 0.6B model has to be trained once (`configs/rerank/gaggle06b_lce.yaml` with
  the released model).

**Cost (extrapolated from pre-training throughput).** The main mix is about 4.4B tokens, about 1.5% of a pre-training
run: about 25 H100-hours for the 3B model (pre-training took ~1,600 for 287B tokens), less for the 0.6B model. The
replay arms range from 3.3B tokens (no replay) to 6.6B (50%). Phases 1–2 need no reranker fine-tuning.

**Rules for every run:**
- Pick no checkpoint by test-set score. Every reranker row uses its final checkpoint, and every CPT run is used at its
  final step.
- Judge reranker differences against run-to-run noise: four trainings of the paper's recipe span 0.638–0.641 on the
  DL mean and 0.542–0.545 on the BEIR mean. Treat a difference under 0.005 on a mean as noise.
- Report the BEIR mean over all 7 collections, as the paper does.

---

## 7. What to report

For every CPT arm, one row:
- **From `results/<run>/{init,step_N}.json`:** held-out corpus bpb, seen-corpus bpb and their difference
  (memorisation), held-out ClimbMix bpb, and CORE.
- **From `eval/results/` for the phase 3–4 arms:** nDCG@10 per collection, the DL mean and the BEIR mean.

Label every reranker number trained from a CPT model as corpus-aware, except, under `exclude_ood`, the four
out-of-domain collections.

## 8. Risks

- **No gain.** The reranker fine-tune (one epoch of 250K query groups) may wash out what CPT adds. The
  memorisation measure (seen vs. held-out bpb) tells "CPT did not take" apart from "it took but did not help".
- **Forgetting hurts BEIR more than exposure helps.** This is what the replay sweep and the CORE gate are for.
- **Robust04 and TREC-News dominate the BEIR group** (~79% of its tokens). If the five smaller collections gain
  nothing, the group ratio and their passes are the levers (with diminishing returns beyond about 4 passes;
  Muennighoff et al.).
- **The 0.6B model may not predict the 3B model.** Phase 4 tests the chosen settings on the 3B model directly.

## References

- Ibrahim et al. 2024. Simple and Scalable Strategies to Continually Pre-train Large Language Models. arXiv:2403.08763.
- Gupta et al. 2023. Continual Pre-Training of Large Language Models: How to (re)warm your model? arXiv:2308.04014.
- Guo et al. 2024. Efficient Continual Pre-training by Mitigating the Stability Gap. arXiv:2406.14833.
- Parmar et al. 2024. Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models. arXiv:2407.07263.
- Bethune et al. 2025. Scaling Laws for Forgetting during Finetuning with Pretraining Data Injection. arXiv:2502.06042.
- EvoLM, 2025. arXiv:2506.16029.
- Abbes et al. 2025. Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models.
  arXiv:2508.01908.
- Que et al. 2024. D-CPT Law: Domain-specific Continual Pre-Training Scaling Law. arXiv:2406.01375.
- Gu et al. 2024. CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training. arXiv:2407.17467.
- Muennighoff et al. 2023. Scaling Data-Constrained Language Models. arXiv:2305.16264.
- Rozière et al. 2023 (Code Llama), Azerbayev et al. 2023 (Llemma), Shao et al. 2024 (DeepSeekMath), Wu et al. 2023
  (PMC-LLaMA), Chen et al. 2023 (Meditron).
- Gao and Callan 2022 (coCondenser), Yu et al. 2022 (COCO-DR), Wang et al. 2022 (GPL), Li et al. 2023 (Llama2Vec).
