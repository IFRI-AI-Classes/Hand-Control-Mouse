import csv

from extraction.csv_io import AtomicCsvWriter
from extraction.csv_schema import COLUMNS
from extraction.dataset_merger import merge_dataset


def _write(path, n_rows, header=COLUMNS):
    with AtomicCsvWriter(path, header) as writer:
        for i in range(n_rows):
            writer.write_row([i] * len(header))


def test_fusion_et_en_tete_inattendu(tmp_path):
    _write(tmp_path / "p01" / "drag" / "a.csv", 3)
    _write(tmp_path / "p02" / "scroll" / "b.csv", 2)
    _write(tmp_path / "p03" / "neutre" / "c.csv", 4, header=["x", "y"])
    total = merge_dataset(tmp_path, tmp_path / "dataset.csv")
    assert total == 5
    with open(tmp_path / "dataset.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == COLUMNS and len(rows) == 6
