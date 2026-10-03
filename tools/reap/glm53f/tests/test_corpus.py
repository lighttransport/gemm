import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

from glm_reap.corpus import load_source, stack_files, normalize, trajectory_rows, download, windows, STACK_REVISION


class SourceLoadingTests(unittest.TestCase):
    def test_existing_byte_cap_writes_manifest_without_loading_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            (dest / "code.jsonl").write_text(json.dumps({"input_ids": [1, 2, 3]})+"\n")
            config = {"corpus_limit_gib": 0, "seed": 42, "code_languages": ["python"], "vision_subsets": []}
            with patch("transformers.AutoTokenizer.from_pretrained") as tokenizer:
                download(config, dest)
            tokenizer.assert_not_called()
            manifest = json.loads((dest / "manifest.json").read_text())
            self.assertTrue(manifest["budget_reached"])
            self.assertEqual(manifest["tokens"], {"code": 3})

    def test_window_mix_and_lazy_vision_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            rows = []
            for kind, count, marker in (("instructions", 100, 1), ("vision", 1000, 2)):
                for i in range(count):
                    rows.append({"kind": kind, "split": "train", "images": ["image.png"] if kind == "vision" else [], "input_ids": [marker]*16, "answer_mask": [False]+[True]*15})
            path.write_text("".join(json.dumps(row)+"\n" for row in rows))
            config = {"source": "unused", "seed": 42, "corpus_mix": {"instructions": .8, "vision": .2}}
            def prepare(row, *args, **kwargs):
                return row["input_ids"], row["answer_mask"], {}
            with patch("transformers.AutoProcessor.from_pretrained"), patch("glm_reap.corpus.vision_window", side_effect=prepare) as vision:
                selected = list(windows(directory, 16, limit=50, config=config))
            instruction_count = sum(row[0][0] == 1 for row in selected)
            self.assertGreater(instruction_count, 30)
            self.assertEqual(vision.call_count, 50-instruction_count)
            self.assertLess(vision.call_count, 20)

    def test_stack_reads_json_files_without_a_dataset_script(self):
        languages = ["python", "c", "c++", "javascript", "typescript", "go", "rust", "java"]
        with patch("datasets.load_dataset") as loader:
            result = load_source("bigcode/the-stack-smol-xs", None, "code", {"code_languages": languages})
        self.assertIs(result, loader.return_value)
        positional, options = loader.call_args
        self.assertEqual(positional, ("json",))
        self.assertTrue(options["streaming"])
        self.assertEqual(options["split"], "train")
        urls = options["data_files"]["train"]
        self.assertEqual(len(urls), len(languages))
        for language, url in zip(languages, urls):
            self.assertIn(STACK_REVISION, url)
            self.assertEqual(url, f"hf://datasets/bigcode/the-stack-smol-xs@{STACK_REVISION}/data/{language}/data.json")
            self.assertNotIn("%2B", url)
        self.assertNotIn("trust_remote_code", options)

    def test_other_sources_keep_their_builtin_configuration(self):
        with patch("datasets.load_dataset") as loader:
            load_source("HuggingFaceFW/fineweb-edu", "sample-10BT", "general", {})
        loader.assert_called_once_with("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    def test_builtin_reader_streams_multiple_language_files(self):
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for language in ("python", "c++"):
                path = Path(directory)/(language+".json")
                path.write_text(json.dumps({"lang": language, "content": "example source"})+"\n")
                files.append(str(path))
            with patch("glm_reap.corpus.stack_files", return_value=files), patch("datasets.config.HF_DATASETS_CACHE", Path(directory)/"cache"):
                data = load_source("bigcode/the-stack-smol-xs", None, "code", {"code_languages": ["python", "c++"]})
                rows = list(data)
            self.assertEqual([row["lang"] for row in rows], ["python", "c++"])
            self.assertEqual([row["content"] for row in rows], ["example source"]*2)

    def test_cpp_filename_survives_hf_filesystem_resolution(self):
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        with patch.object(fs, "_repo_and_revision_exist", return_value=(True, None)):
            path = fs.resolve_path(stack_files({"code_languages": ["c++"]})[0])
        self.assertEqual(path.path_in_repo, "data/c++/data.json")

    def test_trajectory_normalizes_nullable_calls_without_mutating_input(self):
        row = {"trajectory": [{"role": "system", "content": "Use the tools", "tool_calls": None}, {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path":"main.py"}'}}]}]}
        messages = normalize(row, "trajectories")
        self.assertNotIn("tool_calls", messages[0])
        self.assertEqual(messages[1]["content"], "")
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], {"path": "main.py"})
        self.assertIsNone(row["trajectory"][0]["tool_calls"])
        self.assertEqual(row["trajectory"][1]["tool_calls"][0]["function"]["arguments"], '{"path":"main.py"}')

    def test_sync_parquet_reader_can_close_early_and_resume(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory)/"train.parquet"
            data = [{"trajectory_id": str(i), "trajectory": [{"role": "system", "content": "Test", "tool_calls": None}], "unused": "not needed"} for i in range(20)]
            pq.write_table(pa.Table.from_pylist(data), filename, row_group_size=10)
            builder = SimpleNamespace(config=SimpleNamespace(data_files={"train": [str(filename)]}))
            with patch("datasets.load_dataset_builder", return_value=builder):
                rows = trajectory_rows("fixture", None)
                first = next(rows)
                self.assertNotIn("unused", first)
                self.assertEqual(normalize(first, "trajectories")[0]["role"], "system")
                rows.close()
                all_rows = list(trajectory_rows("fixture", None))
            self.assertEqual(len(all_rows), 20)
