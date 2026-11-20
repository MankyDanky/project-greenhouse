"""Smoke test for the bidirectional MNTP continuation, on one GPU with the SDPA path.
1) SDPA non-causal band == dense reference; 2) bidirectional flag makes future tokens influence earlier
positions (and causal does not); 3) MNTP masks never expose a target; 4) a short fresh-optimizer MNTP run
at the production learning rates reduces held-out masked loss on real ClimbMix text."""
import argparse, json, math, os, sys, time
import torch, torch.nn.functional as F
p = argparse.ArgumentParser()
p.add_argument("--src", required=True); p.add_argument("--trainer", required=True)
p.add_argument("--ckpt", required=True); p.add_argument("--step", type=int, required=True)
p.add_argument("--shard", required=True); p.add_argument("--steps", type=int, default=60)
p.add_argument("--dbs", type=int, default=2); p.add_argument("--out", required=True)
a = p.parse_args()
os.environ["NANOCHAT_BIDIRECTIONAL"] = "1"; os.environ["NANOCHAT_BASE_DIR"] = a.ckpt
sys.path.insert(0, a.src); sys.path.insert(0, a.trainer)
import nanochat.gpt as G
import nanochat.flash_attention as FA
from nanochat.checkpoint_manager import build_model
from bidir_mntp import make_mntp
dev = torch.device("cuda"); R = {}

# 1) SDPA band vs dense reference
g = torch.Generator(device=dev); g.manual_seed(0)
q, k, v = (torch.randn(1, 4, 300, 64, device=dev, generator=g) for _ in range(3))
for w in ((32, 32), (-1, -1), (299, 299)):
    y = FA._sdpa_attention(q, k, v, w, False, causal=False)
    i = torch.arange(300, device=dev)[:, None]; j = torch.arange(300, device=dev)[None]
    band = torch.ones(300, 300, dtype=torch.bool, device=dev)
    if 0 <= w[0] < 300: band &= (i - j) <= w[0]
    if 0 <= w[1] < 300: band &= (j - i) <= w[1]
    att = (q @ k.transpose(-1, -2) / math.sqrt(64)).masked_fill(~band, float("-inf")).softmax(-1)
    err = (y - att @ v).abs().max().item(); R[f"sdpa_band_{w}"] = err
    assert err < 1e-4, (w, err)
print("1) SDPA non-causal bands match dense reference", {k_: f"{v_:.1e}" for k_, v_ in R.items()}, flush=True)

R["use_fa3"] = bool(FA.USE_FA3)
if FA.USE_FA3:
    g2 = torch.Generator(device=dev); g2.manual_seed(1)
    q3, k3, v3 = (torch.randn(2, 1024, 4, 128, device=dev, dtype=torch.bfloat16, generator=g2) for _ in range(3))
    for w in ((512, 512), (2048, 2048)):
        y1 = FA._fa3.flash_attn_func(q3, k3, v3, causal=False, window_size=w); y1 = y1[0] if isinstance(y1, tuple) else y1
        y2 = FA._sdpa_attention(q3.transpose(1, 2), k3.transpose(1, 2), v3.transpose(1, 2), w, False, causal=False).transpose(1, 2)
        err = (y1.float() - y2.float()).abs().max().item(); R[f"fa3_vs_sdpa_{w}"] = err
        assert err < 3e-2, (w, err)
    print("1b) FA3 non-causal windows match SDPA reference", {k_: f"{v_:.1e}" for k_, v_ in R.items() if k_.startswith("fa3")}, flush=True)
else:
    print("1b) FA3 not available on this GPU; SDPA path only", flush=True)
model, tok, meta = build_model(a.ckpt, a.step, dev, phase="eval")
BOS = tok.get_bos_token_id(); MASK = tok.encode_special("<|output_end|>")
print(f"loaded step {a.step}; BOS={BOS} MASK={MASK} windows={model.window_sizes[:4]}...", flush=True)

# data: pack real ClimbMix docs (BOS-prefixed) into 2049-token rows
import pyarrow.parquet as pq
texts = pq.read_table(a.shard, columns=["text"]).column("text").to_pylist()
ids = []
for t_ in texts:
    ids += [BOS] + tok.encode(t_)
    if len(ids) > 2049 * (a.dbs * (a.steps + 8) + 8): break
rows = torch.tensor(ids[: (len(ids) // 2049) * 2049]).view(-1, 2049)
X, Y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
print(f"rows {rows.shape[0]} from {a.shard}", flush=True)

# 2) bidirectionality: perturb token 1200, look at position 1000 (outside no window: last layer is full)
with torch.no_grad():
    x = X[:1].to(dev); x2 = x.clone(); x2[0, 1200] = (x2[0, 1200] + 17) % 32000
    G.BIDIRECTIONAL = True
    d_bi = (model(x)[0, 1000] - model(x2)[0, 1000]).abs().max().item()
    G.BIDIRECTIONAL = False
    d_ca = (model(x)[0, 1000] - model(x2)[0, 1000]).abs().max().item()
    G.BIDIRECTIONAL = True
R["future_effect_bidir"], R["future_effect_causal"] = d_bi, d_ca
assert d_ca <= 1e-5 and d_bi > 1e-2 and d_bi > 1000 * max(d_ca, 1e-6), (d_bi, d_ca)
print(f"2) future token changes earlier logits: bidirectional {d_bi:.3e}, causal {d_ca:.1e}", flush=True)

# 3) MNTP structure: every target's own input position is the mask token; targets are the original tokens
mntp = make_mntp(MASK, BOS, 0.2, 1234, 0)
x, y = X[:4].to(dev), Y[:4].to(dev)
xi, tg = mntp(x, y, 0, 0)
sel = tg[:, :-1] != -1
assert torch.all(xi[:, 1:][sel] == MASK) and torch.all(tg[:, :-1][sel] == x[:, 1:][sel])
assert torch.all(xi[x == BOS] == BOS) and (tg[:, -1] == -1).all()
frac = (xi == MASK).float().mean().item(); R["mask_fraction"] = frac
print(f"3) MNTP masks consistent; masked fraction {frac:.3f}", flush=True)

# 4) short fresh-optimizer MNTP run at the production learning rates
model.train()
opt = model.setup_optimizer(unembedding_lr=0.008 * 0.5**0.5, embedding_lr=0.3 * 0.5**0.5, scalar_lr=0.5 * 0.5**0.5,
                            matrix_lr=0.02 * 0.5**0.5, weight_decay=0.0)
lrm_sched = 0.263  # production LR multiplier at the origin step (canonical 282,000)
Xv, Yv = X[-8:].to(dev), Y[-8:].to(dev)
@torch.no_grad()
def val():
    model.eval(); tot = n = c = 0.0
    for b in range(0, 8, 2):
        xi_, tg_ = mntp(Xv[b:b+2], Yv[b:b+2], 10**9, b)
        lg = model(xi_); s_ = tg_ != -1
        tot += F.cross_entropy(lg[s_].float(), tg_[s_], reduction="sum").item(); c += (lg[s_].argmax(-1) == tg_[s_]).sum().item(); n += s_.sum().item()
    model.train(); return tot / n, c / n
v0 = val(); R["val_before"] = v0; print(f"4) held-out MNTP before: loss {v0[0]:.4f} acc {v0[1]:.3f}", flush=True)
t0 = time.time(); losses = []
for s in range(a.steps):
    xb, yb = X[s * a.dbs:(s + 1) * a.dbs].to(dev), Y[s * a.dbs:(s + 1) * a.dbs].to(dev)
    xi_, tg_ = mntp(xb, yb, s, 0)
    loss = model(xi_, tg_); loss.backward()
    lrm = lrm_sched * min(1.0, (s + 1) / 20)
    for gr in opt.param_groups:
        gr["lr"] = gr["initial_lr"] * lrm
        if gr["kind"] == "muon": gr["momentum"] = 0.92
    opt.step(); model.zero_grad(set_to_none=True); losses.append(loss.item())
    assert math.isfinite(losses[-1])
    if s % 10 == 0 or s == a.steps - 1: print(f"   step {s:3d} train MNTP loss {losses[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
v1 = val(); R["val_after"] = v1; R["train_losses"] = losses; R["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
print(f"4) held-out MNTP after {a.steps} steps: loss {v1[0]:.4f} acc {v1[1]:.3f}; peak mem {R['peak_mem_gb']:.1f} GB", flush=True)
assert v1[0] < v0[0], "MNTP loss did not improve"
json.dump(R, open(a.out, "w"), indent=2); print("SMOKE_PASSED", flush=True)
