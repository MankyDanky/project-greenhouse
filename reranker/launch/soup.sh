#!/bin/bash
# The released reranker: an equal-weight average of the four Sec. 5 runs. Run from reranker/.
#
#   bash launch/soup.sh                     # uses checkpoints/sft_run{1,2,3,4}*/
#
# soup.py refuses checkpoints that differ in keys, shapes, prompt, readout or backbone. Run 4 comes from the
# multi engine; its exported model.pt is averaged like the others.
set -euo pipefail
RUN4=${RUN4:-$(ls checkpoints/sft_run4_seed17_gradcache/milestones/step01931_*.pt 2>/dev/null | head -1)}
python soup.py \
  --ckpt checkpoints/sft_run1_seed17/ckpt_final.pt \
  --ckpt checkpoints/sft_run2_seed18/ckpt_final.pt \
  --ckpt checkpoints/sft_run3_seed17_exact_budget/ckpt_final.pt \
  --ckpt "${RUN4:?set RUN4=<run 4 final checkpoint>}" \
  --out checkpoints/sft_soup/ckpt_final.pt
