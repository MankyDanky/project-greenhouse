# Benchmaxxing

Experiments that deliberately adapt Gaggle models to the collections they are evaluated on, to measure how much
that exposure is worth. Each one lives in its own folder with its own plan, configs and scripts, and never changes
what the paper's rows reproduce.

| Folder | Question | Status |
|---|---|---|
| [`cpt/`](cpt/) | Does continued pretraining on the raw evaluation corpora (plus replay) improve the reranker trained afterwards? | iteration 1: plan and scripts ready, CPU-tested |

## Rules for every experiment here

- **Documents only.** Training may use the text of the evaluation corpora. It may never use test queries, relevance
  judgements, candidate lists, or any selection by query.
- **Label the results.** Models trained on an evaluation corpus are *corpus-aware*. Never report their BEIR numbers
  as zero-shot or compare them unmarked with zero-shot rerankers.
- **No checkpoint picking on test sets.** Each row uses its final checkpoint. Intermediate checkpoints are judged on
  held-out text, never on the test collections.
- **Licensed corpora stay private.** Robust04 and TREC-News text may not be redistributed. Do not publish data built
  from them, or a model trained on them, without checking the licence.
- **Separate results.** Results of these experiments go in their own folders, not in `paper/`.
