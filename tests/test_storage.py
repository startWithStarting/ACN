import json
import os
import tempfile
import unittest

from src.storage.ingest import default_run_id, iter_jsonl, load_manifest


class TestStorageIngestUtilities(unittest.TestCase):
    def test_default_run_id_uses_directory_basename(self):
        self.assertEqual(default_run_id("/tmp/acn/run_123"), "run_123")

    def test_iter_jsonl_reads_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rows.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"a": 1}) + "\n")
                f.write(json.dumps({"b": 2}) + "\n")

            self.assertEqual(list(iter_jsonl(path)), [{"a": 1}, {"b": 2}])

    def test_load_manifest_reads_trace_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = os.path.join(tmpdir, "trace")
            os.makedirs(trace_dir)
            manifest = {"schema_version": 1, "mode": "parallel"}
            with open(os.path.join(trace_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            self.assertEqual(load_manifest(tmpdir), manifest)


if __name__ == "__main__":
    unittest.main()
