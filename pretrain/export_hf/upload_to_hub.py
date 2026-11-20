#!/usr/bin/env python
"""Add the transformers / vLLM files to the existing model repo in ONE commit, leaving the nanochat files untouched.

    python upload_to_hub.py --dir HF_DIR --repo castorini/gaggle-nanochat-pretrained-climbmix-20260924 \
        --parent 363ba278c2ef13f2997a4d97df793b1dd327efda [--dry-run]

`--parent` makes the commit fail if the repo changed since it was inspected. SHA256SUMS is regenerated to cover
the original files (whose checksums must still match) and every added file.
"""
import argparse, hashlib, json, os

ADD = ["config.json", "generation_config.json", "configuration_gaggle.py", "modeling_gaggle.py",
       "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "README.md",
       "convert_nanochat_to_hf.py", "vllm_gaggle-0.1.0-py3-none-any.whl",
       "vllm_plugin/pyproject.toml", "vllm_plugin/vllm_gaggle/__init__.py", "vllm_plugin/vllm_gaggle/model.py"]
ORIGINAL = {"model_227158.pt": "0d314c2a3441fc81fbecb0e58e9040b6758a1094afbb35621c059aca1f30baf6",
            "meta_227158.json": "1a5edd78716655f5156c63c4be9b988ed1335ce043b27ae0bc1e2e1afd2d97ad"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()

    shards = sorted(json.load(open(os.path.join(a.dir, "model.safetensors.index.json")))["weight_map"].values())
    files = sorted(set(shards)) + ADD
    missing = [f for f in files if not os.path.exists(os.path.join(a.dir, f))]
    assert not missing, f"missing files: {missing}"

    info = api.model_info(a.repo, files_metadata=True)
    assert info.sha == a.parent, f"repo moved: head {info.sha} != expected parent {a.parent}"
    present = {s.rfilename for s in info.siblings}
    clobber = [f for f in files if f in present and f != "README.md"]
    assert not clobber, f"would overwrite existing files: {clobber}"
    for name, want in ORIGINAL.items():
        lfs = next(s for s in info.siblings if s.rfilename == name)
        got = lfs.lfs.sha256 if lfs.lfs else None
        assert got in (None, want), f"{name} on the hub has sha {got}, expected {want}"

    sums = dict(ORIGINAL)
    for f in files:
        if f != "README.md":
            sums[f] = sha256(os.path.join(a.dir, f))
    with open(os.path.join(a.dir, "SHA256SUMS"), "w") as fh:
        for f in sorted(sums):
            fh.write(f"{sums[f]}  {f}\n")
    files.append("SHA256SUMS")

    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=os.path.join(a.dir, f)) for f in files]
    print(json.dumps({"repo": a.repo, "parent": a.parent, "add": files,
                      "bytes": sum(os.path.getsize(os.path.join(a.dir, f)) for f in files)}, indent=1))
    if a.dry_run:
        return
    c = api.create_commit(a.repo, operations=ops, parent_commit=a.parent,
                          commit_message="Add transformers and vLLM support (same weights, safetensors); nanochat files unchanged",
                          commit_description="Adds a trust_remote_code transformers port (configuration_gaggle.py, "
                          "modeling_gaggle.py), the weights as safetensors (tensors identical to model_227158.pt, "
                          "renamed), tokenizer.json, and a vLLM plugin. The original nanochat checkpoint files are "
                          "unchanged. See README for the parity checks.")
    print("committed", c.oid, c.commit_url)


if __name__ == "__main__":
    main()
