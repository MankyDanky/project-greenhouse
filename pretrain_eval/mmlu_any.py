"""Standard 5-shot MMLU (lm-evaluation-harness `mmlu`) for nanochat or HuggingFace models through one adapter:
HFLM-style context/continuation split, the model's own BOS prepended, left truncation at 2,048 tokens,
one forward per unique context (answer letters are single tokens)."""
import argparse, json, time, torch, torch.nn.functional as F
from models import load
import lm_eval
from lm_eval.api.model import LM

p = argparse.ArgumentParser()
p.add_argument("--kind", required=True); p.add_argument("--src"); p.add_argument("--ckpt"); p.add_argument("--step", type=int)
p.add_argument("--hf"); p.add_argument("--out", required=True); p.add_argument("--limit", type=int)
p.add_argument("--task", default="mmlu", help="lm-eval task, e.g. mmlu or leaderboard_mmlu_pro (both 5-shot, answer-letter log-likelihood)")
p.add_argument("--max-len", type=int, default=2048, help="-1 = the model's own context length"); p.add_argument("--token-budget", type=int, default=32768)
p.add_argument("--mntp-mask-id", type=int, default=-1, help="bidirectional MNTP scoring: the answer tokens are replaced by "
               "this id and each is predicted from the position before it; contexts are grouped by exact length, never padded")
a = p.parse_args()
t0 = time.time()
model, tok, native_ctx, desc = load(a.kind, a.src, a.ckpt, a.step, a.hf)
BOS = tok.get_bos_token_id(); dev = torch.device("cuda"); MAXLEN = a.max_len if a.max_len > 0 else native_ctx
PAD = BOS if BOS is not None else 0; MASK = a.mntp_mask_id; MNTP = MASK >= 0
print(f"loaded {desc} in {time.time()-t0:.0f}s; BOS={BOS}; letters={[tok.encode(' '+L) for L in 'ABCD']}", flush=True)

class Adapter(LM):
    def __init__(self):
        super().__init__(); self.stats = {"requests": 0, "forwards": 0, "truncated": 0}
    def _pair(self, context, continuation):
        n = len(context) - len(context.rstrip())
        if n > 0: continuation, context = context[-n:] + continuation, context[:-n]
        whole = tok.encode(context + continuation); ctx = tok.encode(context)
        return ([BOS] if BOS is not None else []) + ctx, whole[len(ctx):]
    @torch.no_grad()
    def loglikelihood(self, requests, disable_tqdm=False):
        self.stats["requests"] += len(requests); items = []
        for r in requests:
            ctx, cont = self._pair(*r.args); full = ctx + cont
            if MNTP:
                # Input = context + one mask per answer token (answer never visible); output at t predicts t+1.
                if len(full) > MAXLEN: self.stats["truncated"] += 1
                x = full[-MAXLEN:]; c = len(cont)
                items.append((x[:len(x) - c] + [MASK] * c, cont))
                continue
            if len(full) > MAXLEN + 1: self.stats["truncated"] += 1
            items.append((full[-(MAXLEN + 1):][:-1], cont))
        groups = {}
        for i, (inp, cont) in enumerate(items): groups.setdefault(tuple(inp), []).append(i)
        keys = sorted(groups, key=len, reverse=True); res = [None] * len(items); b = 0
        while b < len(keys):
            n = max(1, a.token_budget // max(len(keys[b]), 1)); batch = keys[b:b + n]
            if MNTP:  # equal lengths only: padding would be attended under bidirectional attention
                batch = [k for k in batch if len(k) == len(keys[b])]; n = len(batch)
            T = max(len(k) for k in batch)
            x = torch.full((len(batch), T), PAD, dtype=torch.long, device=dev)
            for j, k in enumerate(batch): x[j, :len(k)] = torch.tensor(k, device=dev)
            logits = model(x); self.stats["forwards"] += len(batch)
            for j, k in enumerate(batch):
                L = len(k)
                for i in groups[k]:
                    cont = items[i][1]; c = len(cont)
                    lo = L - c - 1 if MNTP else L - c
                    rows = F.log_softmax(logits[j, lo:lo + c].float(), -1); tgt = torch.tensor(cont, device=dev)
                    res[i] = (rows.gather(-1, tgt[:, None]).sum().item(), bool((rows.argmax(-1) == tgt).all().item()))
            b += n
        return res
    def loglikelihood_rolling(self, requests, disable_tqdm=False): raise NotImplementedError
    def generate_until(self, requests, disable_tqdm=False): raise NotImplementedError

lm = Adapter(); t1 = time.time()
r = lm_eval.simple_evaluate(model=lm, tasks=[a.task], num_fewshot=5, limit=a.limit, log_samples=False, bootstrap_iters=0)
res = r["results"]; g = res[a.task]["acc,none"]
json.dump({"model": desc, "task": a.task, "max_len": MAXLEN, "mmlu": g, "acc": g, "groups": {k: res[k]["acc,none"] for k in list(("mmlu_humanities","mmlu_social_sciences","mmlu_stem","mmlu_other")) + [c for c in r.get("group_subtasks", {}).get(a.task, []) if isinstance(c, str)] if k in res and "acc,none" in res[k]},
           "stats": lm.stats, "seconds": time.time() - t1, "mntp_mask_id": MASK, "prefix_token": BOS, "lm_eval": lm_eval.__version__, "torch": torch.__version__}, open(a.out, "w"), indent=2)
print(f"{a.task.upper()} {g:.4f} | {desc} | {(time.time()-t1)/60:.1f} min | {lm.stats}", flush=True)
