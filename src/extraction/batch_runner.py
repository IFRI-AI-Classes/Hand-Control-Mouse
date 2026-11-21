"""Orchestration d'un lot : choisit les vidéos, les traite une à une, met à jour le rapport."""
from dataclasses import dataclass
from pathlib import Path

from .dataset_merger import merge_dataset
from .extraction_options import ExtractionOptions
from .hand_extractor import create_mediapipe_extractor
from .preview_renderer import PreviewRenderer
from .quality_checks import collect_warnings
from .report import (
    STATUS_ERROR, STATUS_IGNORED, build_failure_row, build_ok_row, format_summary, update_report,
)
from .video_discovery import find_videos
from .video_labels import LabelError, VideoLabel, parse_video_label
from .video_processor import VideoProcessingError, process_video
from .video_reader import VideoOpenError

EXIT_OK, EXIT_VIDEO_ERRORS, EXIT_BAD_USAGE = 0, 1, 2


@dataclass(frozen=True)
class BatchSettings:
    input_dir: Path
    output_dir: Path
    model_path: Path
    options: ExtractionOptions
    strict: bool = False
    force: bool = False
    merge: bool = False
    only: Path | None = None
    preview: bool = False


def run_batch(settings: BatchSettings, extractor_factory=None) -> int:
    if extractor_factory is None:
        if not Path(settings.model_path).is_file():
            print(f"Erreur : modèle MediaPipe introuvable : {settings.model_path} (option --model)")
            return EXIT_BAD_USAGE
        extractor_factory = lambda: create_mediapipe_extractor(settings.model_path)
    videos = _select_videos(settings)
    if videos is None:
        return EXIT_BAD_USAGE
    rows = _process_all(videos, settings, extractor_factory)
    update_report(Path(settings.output_dir) / "report.csv", rows)
    print(format_summary(rows))
    if settings.merge:
        count = merge_dataset(settings.output_dir, Path(settings.output_dir) / "dataset.csv")
        print(f"dataset.csv : {count} lignes")
    return EXIT_VIDEO_ERRORS if any(r["status"] == STATUS_ERROR for r in rows) else EXIT_OK


def _select_videos(settings: BatchSettings) -> list | None:
    if settings.only is not None:
        only = Path(settings.only)
        if not only.is_file():
            print(f"Erreur : vidéo introuvable : {only}")
            return None
        return [only.resolve()]
    videos = find_videos(settings.input_dir)
    if not videos:
        print(f"Aucune vidéo trouvée dans {settings.input_dir}")
    return videos


def _process_all(videos: list, settings: BatchSettings, extractor_factory) -> list:
    rows = []
    for number, video in enumerate(videos, start=1):
        row = _handle_video(video, settings, extractor_factory, f"[{number}/{len(videos)}]")
        if row is not None:
            rows.append(row)
    return rows


def _handle_video(video: Path, settings: BatchSettings, extractor_factory, prefix: str) -> dict | None:
    name = _display_name(video, settings.input_dir)
    try:
        label = parse_video_label(video, settings.input_dir)
    except LabelError as error:
        return _announce(prefix, build_failure_row(name, STATUS_IGNORED, str(error)))
    if settings.strict and not label.filename_ok:
        return _announce(prefix, build_failure_row(name, STATUS_IGNORED, label.problems[0], label))
    output_csv = _csv_path(label, video, settings.output_dir)
    if output_csv.exists() and not settings.force and not settings.preview:
        print(f"{prefix} {name} : déjà traitée, sautée")
        return None
    return _announce(prefix, _run_video(video, label, output_csv, settings, extractor_factory))


def _run_video(video, label: VideoLabel, output_csv: Path, settings, extractor_factory) -> dict:
    sink = PreviewRenderer(_preview_path(label, video, settings.output_dir)) if settings.preview else None
    try:
        stats = process_video(video, label, extractor_factory, output_csv, settings.options, sink)
    except (VideoOpenError, VideoProcessingError) as error:
        return build_failure_row(label.video, STATUS_ERROR, str(error), label)
    except Exception as error:  # une vidéo cassée ne doit jamais arrêter le lot
        return build_failure_row(label.video, STATUS_ERROR, f"{type(error).__name__}: {error}", label)
    return build_ok_row(label, stats, collect_warnings(stats, label))


def _announce(prefix: str, row: dict) -> dict:
    if row["status"] == "ok":
        detail = f"détection {float(row['detection_rate']):.0%}, {row['frames_kept']} images"
    else:
        detail = f"{row['status']} - {row['message']}"
    print(f"{prefix} {row['video']} : {detail}")
    if row["warnings"]:
        print(f"      ! {row['warnings']}")
    return row


def _display_name(video: Path, input_dir: Path) -> str:
    try:
        return Path(video).resolve().relative_to(Path(input_dir).resolve()).as_posix()
    except ValueError:
        return str(video)


def _csv_path(label: VideoLabel, video: Path, output_dir: Path) -> Path:
    return Path(output_dir) / label.person / label.gesture / f"{Path(video).stem}.csv"


def _preview_path(label: VideoLabel, video: Path, output_dir: Path) -> Path:
    return Path(output_dir) / "preview" / f"{label.person}_{label.gesture}_{Path(video).stem}.mp4"
