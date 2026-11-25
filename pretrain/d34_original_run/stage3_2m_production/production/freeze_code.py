"""Freeze this production implementation's content identity, without approval or jobs."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
files = []
for path in sorted(root.iterdir()):
    if path.suffix in (".py", ".sh", ".sbatch") and not path.name.startswith("test_"):
        files.append({"path": path.name, "size": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
manifest = root / "code_manifest.json"
manifest.write_text(json.dumps({"format": "d34-production-code-v1", "files": files}, indent=2, sort_keys=True) + "\n")
print(json.dumps({"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  "files": len(files)}, sort_keys=True))
