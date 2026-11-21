"""Rapport par vidéo : construction des lignes, mise à jour de report.csv, résumé console."""
import csv
from pathlib import Path

from .video_labels import VideoLabel
from .video_stats import VideoStats

STATUS_OK = "ok"
STATUS_IGNORED = "ignoree"
STATUS_ERROR = "erreur"

REPORT_COLUMNS = [
    "video", "status", "message", "person", "gesture", "lighting", "orientation", "source",
    "frames_read", "frames_kept", "duration_s", "fps", "width", "height",
    "trim_start_s", "trim_end_s", "detection_rate", "majority_side", "majority_side_share", "warnings",
]


def build_ok_row(label: VideoLabel, stats: VideoStats, warnings: list) -> dict:
    return {
        "video": label.video, "status": STATUS_OK, "message": "",
        "person": label.person, "gesture": label.gesture, "lighting": label.lighting,
        "orientation": label.orientation, "source": label.source,
        "frames_read": stats.frames_read, "frames_kept": stats.frames_kept,
        "duration_s": f"{stats.duration_s:.2f}", "fps": f"{stats.fps:.2f}",
        "width": stats.width, "height": stats.height,
        "trim_start_s": stats.trim_start_s, "trim_end_s": stats.trim_end_s,
        "detection_rate": f"{stats.detection_rate:.3f}",
        "majority_side": stats.majority_side, "majority_side_share": f"{stats.majority_side_share:.3f}",
        "warnings": " ; ".join(warnings),
    }


def build_failure_row(video: str, status: str, message: str, label: VideoLabel | None = None) -> dict:
    row = {column: "" for column in REPORT_COLUMNS}
    row.update(video=video, status=status, message=message)
    if label is not None:
        row.update(person=label.person, gesture=label.gesture, lighting=label.lighting,
                   orientation=label.orientation, source=label.source)
    return row


def update_report(report_path: Path, new_rows: list) -> None:
    """Fusionne les nouvelles lignes avec le rapport existant (clé : chemin de la vidéo)."""
    merged = {row["video"]: row for row in _read_existing(report_path)}
    merged.update({row["video"]: row for row in new_rows})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged[key] for key in sorted(merged))


def _read_existing(report_path: Path) -> list:
    if not Path(report_path).is_file():
        return []
    with open(report_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_summary(rows: list) -> str:
    counts = {status: sum(1 for r in rows if r["status"] == status)
              for status in (STATUS_OK, STATUS_IGNORED, STATUS_ERROR)}
    flagged = sum(1 for r in rows if r["status"] == STATUS_OK and r["warnings"])
    return (f"Résumé : {counts[STATUS_OK]} traitée(s) dont {flagged} avec avertissement(s), "
            f"{counts[STATUS_IGNORED]} ignorée(s), {counts[STATUS_ERROR]} en erreur.")
