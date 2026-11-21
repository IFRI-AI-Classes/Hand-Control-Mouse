"""Traitement d'une vidéo : lecture, miroir, détection, écriture du CSV."""
from pathlib import Path

import cv2

from .csv_io import AtomicCsvWriter
from .csv_schema import COLUMNS, build_row
from .data_models import FrameRecord
from .extraction_options import ExtractionOptions
from .timeline import build_keep_window, frame_timestamp_ms
from .video_labels import VideoLabel
from .video_reader import VideoSource
from .video_stats import VideoStats


class VideoProcessingError(Exception):
    """La vidéo a été lue mais ne donne aucune image exploitable."""


def process_video(video_path, label: VideoLabel, extractor_factory, output_csv, options: ExtractionOptions,
                  sink=None) -> VideoStats:
    stats = VideoStats(trim_start_s=options.trim_start_s, trim_end_s=options.trim_end_s)
    extractor = extractor_factory()
    try:
        with VideoSource(video_path) as source:
            _fill_video_info(stats, source.info)
            window = build_keep_window(source.info.frame_count, source.info.fps,
                                       options.trim_start_s, options.trim_end_s)
            stats.trim_end_applied = options.trim_end_s <= 0 or window.end_ms is not None
            if sink:
                sink.start(source.info)
            with AtomicCsvWriter(output_csv, COLUMNS) as writer:
                for frame_idx, frame in source.frames():
                    stats.frames_read += 1
                    t_ms = frame_timestamp_ms(frame_idx, source.info.fps)
                    if window.contains(t_ms):
                        _handle_frame(frame, frame_idx, t_ms, label, extractor, options, writer, stats, sink)
                if stats.frames_kept == 0:
                    raise VideoProcessingError("aucune image exploitable (vidéo vide ou trop courte pour le trim)")
            stats.duration_s = stats.frames_read / source.info.fps
    finally:
        extractor.close()
        if sink:
            sink.close()
    return stats


def _fill_video_info(stats: VideoStats, info) -> None:
    stats.fps = info.fps
    stats.fps_estimated = info.fps_estimated


def _handle_frame(frame, frame_idx, t_ms, label, extractor, options, writer, stats, sink) -> None:
    mirrored = cv2.flip(frame, 1) if options.mirror else frame
    rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
    observation = extractor.extract(rgb, t_ms)
    height, width = mirrored.shape[:2]
    record = FrameRecord(frame_idx, t_ms, width, height)
    writer.write_row(build_row(label, record, observation))
    stats.add_frame(record, observation)
    if sink:
        sink.on_frame(mirrored, label, record, observation)
