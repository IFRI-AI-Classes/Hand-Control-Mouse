"""Écriture d'un CSV « tout ou rien » : le fichier final n'apparaît que si l'écriture a réussi."""
import csv
import os
from pathlib import Path


class AtomicCsvWriter:
    def __init__(self, final_path, header):
        self._final = Path(final_path)
        self._tmp = self._final.with_name(self._final.name + ".tmp")
        self._header = header
        self._file = None
        self._writer = None

    def __enter__(self):
        self._final.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._tmp, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, lineterminator="\n")
        self._writer.writerow(self._header)
        return self

    def write_row(self, row):
        self._writer.writerow(row)

    def __exit__(self, exc_type, exc, traceback):
        self._file.close()
        if exc_type is None:
            os.replace(self._tmp, self._final)
        else:
            self._tmp.unlink(missing_ok=True)
        return False
