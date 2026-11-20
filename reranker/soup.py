#!/usr/bin/env python
"""Equal-weight average (model soup) of several reranker checkpoints.

Averaging only makes sense for fine-tunes of the SAME pretrained initialisation, which these are
(all start from nanochat-4b-climbmix-0924). The script refuses anything else it can detect: a key
set mismatch, a shape mismatch, or a differing prompt/readout/backbone in the saved cfg, because a
soup across those would produce a model that is not meaningfully any of its parents.

Accumulates in float64 so the mean of three fp32 tensors does not lose low bits, and writes a
model+cfg checkpoint (no optimizer state), which is what evaluation reads.
"""
import argparse, json, os, torch

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', action='append', required=True, help='repeat once per checkpoint')
ap.add_argument('--out', required=True)
ap.add_argument('--cfg-from', default=None, help='which checkpoint supplies cfg (default: the first)')
a = ap.parse_args()

print(f'souping {len(a.ckpt)} checkpoints, equal weights {1/len(a.ckpt):.4f} each', flush=True)
acc, keys, ref_cfg, n = None, None, None, 0
for path in a.ckpt:
    sd = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    model, cfg = sd['model'], dict(sd.get('cfg') or {})
    step = sd.get('step')
    if keys is None:
        keys = set(model)
        ref_cfg = cfg
        acc = {k: model[k].to(torch.float64) for k in model}
    else:
        if set(model) != keys:
            raise SystemExit(f'{path}: key set differs ({len(set(model) ^ keys)} keys differ)')
        for k in keys:
            if model[k].shape != acc[k].shape:
                raise SystemExit(f'{path}: shape mismatch at {k}: {tuple(model[k].shape)}')
            acc[k] += model[k].to(torch.float64)
        for field in ('model_name', 'prompt', 'score_mode', 'bidirectional', 'backbone_impl'):
            mine, ref = cfg.get(field), ref_cfg.get(field)
            if field == 'model_name':
                # the same backbone can be named relatively or absolutely, and the recipe's
                # models/ entries may be symlinks into a shared copy -- compare what the paths
                # actually resolve to, falling back to the basename for a path that is not present
                def norm(v):
                    # a checkpoint trained elsewhere names a backbone path that does not exist here
                    # (R0 came from Alliance Rorqual), so return both the resolved path and the
                    # basename and accept a match on either
                    if not isinstance(v, str):
                        return (v, v)
                    full = v if os.path.isabs(v) else os.path.join(os.getcwd(), v)
                    real = os.path.realpath(full) if os.path.exists(full) else None
                    return (real, os.path.basename(v.rstrip('/')))
                a_real, a_base = norm(mine)
                b_real, b_base = norm(ref)
                if (a_real and b_real and a_real == b_real) or a_base == b_base:
                    mine = ref = None          # same backbone, differently spelled
                else:
                    mine, ref = mine, ref
            if mine != ref:
                raise SystemExit(f'{path}: cfg.{field}={mine!r} but the first has {ref!r}; '
                                 f'these are not the same model family')
    n += 1
    print(f'  + {path}  (step {step}, {len(model)} tensors)', flush=True)
    del sd, model

out_model = {k: (acc[k] / n).to(torch.float32) for k in acc}
# a soup should sit near its parents, not somewhere unrelated: report the drift on a few tensors
probe = [k for k in sorted(out_model) if out_model[k].dim() >= 2][:3]
first = torch.load(a.ckpt[0], map_location='cpu', weights_only=False, mmap=True)['model']
for k in probe:
    d = (out_model[k] - first[k].to(torch.float32)).norm() / first[k].to(torch.float32).norm()
    print(f'  drift vs first parent, {k}: {d:.4%}', flush=True)

cfg_src = a.cfg_from or a.ckpt[0]
sd_cfg = torch.load(cfg_src, map_location='cpu', weights_only=False, mmap=True)
payload = {'model': out_model, 'cfg': sd_cfg['cfg'], 'step': sd_cfg.get('step'),
           'soup': {'n': n, 'weights': 'equal', 'sources': a.ckpt, 'cfg_from': cfg_src}}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
tmp = a.out + '.tmp'
torch.save(payload, tmp)
os.replace(tmp, a.out)
print(f'wrote {a.out}  ({os.path.getsize(a.out)/2**30:.1f} GiB)', flush=True)
print(json.dumps(payload['soup'], indent=2))
