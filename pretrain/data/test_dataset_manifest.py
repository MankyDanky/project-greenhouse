"""Filesystem index tests; these do not tokenize documents or run a model."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load_list_function(filename, default_dir):
    # Execute the actual function without importing optional cluster libraries.
    tree = ast.parse((HERE / filename).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "list_parquet_files")
    namespace = {"os": os, "json": json, "DATA_DIR": str(default_dir),
                 "MAX_SHARD": 6542, "index_to_filename": lambda i: f"shard_{i:05d}.parquet"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return namespace["list_parquet_files"]


class IndexedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.full, self.sparse = self.root / "full", self.root / "sparse"
        self.full.mkdir()
        self.sparse.mkdir()
        for i in range(6543):
            (self.full / f"shard_{i:05d}.parquet").touch()
        for line in (HERE / "stage_shards.txt").read_text().splitlines():
            (self.sparse / line).touch()
        self.manifest = self.root / "manifest.json"
        self.manifest.write_bytes((HERE / "indexed_shards.json").read_bytes())
        self.original = load_list_function("dataset.original.py", self.full)
        self.modified = load_list_function("../nanochat/nanochat/dataset.py", self.sparse)
        self.environment = patch.dict(os.environ, {
            "D34_SHARD_MANIFEST": str(self.manifest), "D34_DATA_ROOT": str(self.sparse)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def test_sparse_directory_preserves_every_original_list_index(self):
        original = self.original()
        indexed = self.modified()
        self.assertEqual([Path(p).name for p in indexed], [Path(p).name for p in original])
        self.assertEqual(Path(indexed[1945]).name, "shard_01945.parquet")
        self.assertEqual(Path(indexed[-1]).name, "shard_06542.parquet")
        self.assertEqual(len(indexed[:-1]), 6542)
        self.assertTrue(Path(indexed[2072]).exists())
        self.assertFalse(Path(indexed[2073]).exists())
        with self.assertRaises(FileNotFoundError):
            open(indexed[2073], "rb")

    def test_unset_manifest_preserves_original_directory_enumeration(self):
        with patch.dict(os.environ):
            del os.environ["D34_SHARD_MANIFEST"]
            self.assertEqual(len(self.modified()), 129)

    def test_reordered_or_shortened_manifest_is_rejected(self):
        value = json.loads(self.manifest.read_text())
        value["parquet_basenames"][1945], value["parquet_basenames"][1946] = (
            value["parquet_basenames"][1946], value["parquet_basenames"][1945])
        self.manifest.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.modified()

    def test_conflicting_directory_argument_is_rejected(self):
        with self.assertRaises(ValueError):
            self.modified(str(self.full))


if __name__ == "__main__":
    unittest.main()
