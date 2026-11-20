# Records of the original run

SHA256 pins from the run that produced the released model (2026-09). Your own run produces its own checkpoints and
manifests; these let you check that the code and released files you have are the ones this recipe describes.

| What | SHA256 |
|---|---|
| `model_227158.pt` (released, also in the HF repo's `SHA256SUMS`) | `0d314c2a3441fc81fbecb0e58e9040b6758a1094afbb35621c059aca1f30baf6` |
| `meta_227158.json` (released) | `1a5edd78716655f5156c63c4be9b988ed1335ce043b27ae0bc1e2e1afd2d97ad` |
| `tokenizer/tokenizer.pkl` (released; stage 0 reproduces it byte for byte) | `ae73c5f7a960edc56022dfd46a653df2b9b38d84456e5e9f48eb5e02a60c21c2` |
| production code manifest ([`production_code_manifest.json`](production_code_manifest.json)) | `c2c83e580d6a57c13c81c5ab750ee5dd1a23b28027c0094b8c38498ea051120a` |
| stage-2 winner checkpoint manifest (ten files of step 137,340) | `563b5127f5ec627423ce8b71d25e1789b2093e8c553a554e3f39f6a56ad54794` |
| stage-2 `COMPLETE.json` | `e1b72912d274b369878de4828b79161dbd6899c32e3e5a203c3d57f4b25a0d53` |
| full-corpus manifest (all 6,543 ClimbMix shards) | `484130e26069c1df33f400b8b1220b6fcbb2100bcb621899ffbaa612664d2ac9` |

`production_code_manifest.json` is the manifest bound by the original production release. Every file it lists is
byte-identical to `../stage3_2m_production/production/` except `fir_config.example.sh`, a configuration template with
our cluster paths that this recipe replaces with `config.sh`; `prepare_release.sbatch` writes a fresh manifest for your
frozen copy. Check:

```bash
cd ../stage3_2m_production/production && python3 -c "
import hashlib, json
for r in json.load(open('../../records/production_code_manifest.json'))['files']:
    try: ok = hashlib.sha256(open(r['path'], 'rb').read()).hexdigest() == r['sha256']
    except FileNotFoundError: ok = None
    print({True: 'same   ', False: 'DIFFERS', None: 'absent '}[ok], r['path'])"
```

Original placement: stages 1 and 3 on one 8xH100 node of the Alliance Killarney cluster in 24-hour legs; stage 2's
1,024 updates on two H200 GPUs (watgpu, University of Waterloo) through an adapter that replays the eight ranks' data
streams and gradient accumulation exactly, then exported to the same eight-rank checkpoint format that
`../stage2_2m_continuation/run_branch.sbatch` writes.
