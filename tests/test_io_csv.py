import tempfile
import unittest
from pathlib import Path

from crpf.io_csv import merge_csv_tree, read_csv_any_encoding, write_csv


class CsvIoTests(unittest.TestCase):
    def test_read_and_write_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            write_csv(path, [{"name": "咲季", "text": "こんにちは"}])
            rows = read_csv_any_encoding(path)
            self.assertEqual(rows[0]["name"], "咲季")

    def test_merge_csv_tree_adds_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "pstory" / "hski"
            nested.mkdir(parents=True)
            write_csv(nested / "a.csv", [{"name": "咲季", "text": "当然よ"}])
            rows = merge_csv_tree(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_kind"], "pstory")
            self.assertTrue(rows[0]["source_file"].endswith("a.csv"))


if __name__ == "__main__":
    unittest.main()
