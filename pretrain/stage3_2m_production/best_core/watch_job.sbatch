#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --job-name=d34-bestcore-watch
#SBATCH --mail-type=NONE
# CPU-only watcher; never touches training jobs. Chains itself until BASE_DONE. Start it with the production chain:
#   bash submit.sh cpu stage3_2m_production/best_core/watch_job.sbatch
set -u
source "${D34_CONFIG:?export D34_CONFIG=/absolute/path/to/config.sh}"
HERE=$D34_RECIPE/stage3_2m_production/best_core
BEST=$D34_ROOT/production/best_core; mkdir -p "$BEST"
PD=$D34_PROD_DIR
next=""
if [[ ! -e "$PD/BASE_DONE" ]]; then
    next=$(sbatch --parsable $D34_CPU_SBATCH --output="$BEST/watch-%j.out" --dependency=afterany:$SLURM_JOB_ID "$HERE/watch_job.sbatch")
    echo "$SLURM_JOB_ID -> $next $(date -Is)" >> "$BEST/watch_jobs.tsv"
fi
"$D34_PYTHON" "$HERE/watch_best_core.py" --prod-dir "$PD" --ckpt-dir "$D34_BASE_DIR/base_checkpoints/$D34_PROD_TAG" \
    --best-dir "$BEST" --loop --interval 300 --max-seconds $((23 * 3600 + 30 * 60))
rc=$?
if [[ $rc == 3 && -n "$next" ]]; then scancel "$next"; echo "cancelled successor $next (BASE_DONE)" >> "$BEST/watch_jobs.tsv"; fi
exit 0
