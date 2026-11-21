"""Fusion de tous les CSV par vidéo en un seul dataset.csv."""
import csv
from pathlib import Path

from .csv_io import AtomicCsvWriter
from .csv_schema import COLUMNS


def merge_dataset(output_dir: Path, dataset_path: Path) -> int:
    """Retourne le nombre de lignes de données écrites. Les CSV à l'en-tête inattendu sont signalés et sautés."""
    total = 0
    with AtomicCsvWriter(dataset_path, COLUMNS) as writer:
        for source in sorted(Path(output_dir).glob("*/*/*.csv")):
            total += _copy_rows(source, writer)
    return total


def _copy_rows(source: Path, writer: AtomicCsvWriter) -> int:
    with open(source, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != COLUMNS:
            print(f"  ! {source.name} : en-tête inattendu, fichier ignoré dans la fusion")
            return 0
        count = 0
        for row in reader:
            writer.write_row(row)
            count += 1
        return count
