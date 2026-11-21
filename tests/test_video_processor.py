import csv

import pytest

from extraction.csv_schema import COLUMNS
from extraction.extraction_options import ExtractionOptions
from extraction.video_labels import VideoLabel
from extraction.video_processor import VideoProcessingError, process_video
from fake_extractor import FakeExtractor, factory
from video_builder import make_video

LABEL = VideoLabel("p01", "drag", "jour", "face", "webcam", "p01/drag/jour_face_webcam.avi", True)
OPTIONS = ExtractionOptions(trim_start_s=0.5, trim_end_s=0.5)


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_lignes_apres_trim_et_76_colonnes(tmp_path):
    video = make_video(tmp_path / "v.avi", n_frames=30, fps=10)
    out = tmp_path / "out.csv"
    stats = process_video(video, LABEL, factory(), out, OPTIONS)
    rows = _read(out)
    assert rows[0] == COLUMNS and all(len(r) == 76 for r in rows)
    assert len(rows) - 1 == 21 == stats.frames_kept and stats.frames_read == 30


def test_timestamps_strictement_croissants(tmp_path):
    video = make_video(tmp_path / "v.avi")
    extractor = FakeExtractor()
    process_video(video, LABEL, lambda: extractor, tmp_path / "out.csv", OPTIONS)
    assert extractor.timestamps == sorted(set(extractor.timestamps)) and extractor.timestamps[0] == 500
    assert extractor.closed


def test_images_sans_main_ecrites(tmp_path):
    video = make_video(tmp_path / "v.avi")
    out = tmp_path / "out.csv"
    stats = process_video(video, LABEL, factory(missing_every=3), out, OPTIONS)
    rows = _read(out)[1:]
    assert stats.frames_detected < stats.frames_kept == len(rows)
    assert {r[10] for r in rows} == {"0", "1"}
    assert all(set(r[11:]) == {""} for r in rows if r[10] == "0")


def test_video_trop_courte_pour_le_trim_ne_laisse_aucun_csv(tmp_path):
    video = make_video(tmp_path / "v.avi", n_frames=5, fps=10)  # 0,5 s de vidéo, trim de 1 s au total
    out = tmp_path / "out.csv"
    with pytest.raises(VideoProcessingError):
        process_video(video, LABEL, factory(), out, OPTIONS)
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))


def test_erreur_du_detecteur_ne_laisse_pas_de_csv_partiel(tmp_path):
    class Boom(FakeExtractor):
        def extract(self, rgb_frame, timestamp_ms):
            raise RuntimeError("boom")

    video = make_video(tmp_path / "v.avi")
    out = tmp_path / "out.csv"
    with pytest.raises(RuntimeError):
        process_video(video, LABEL, lambda: Boom(), out, OPTIONS)
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))


def test_miroir_applique_au_detecteur(tmp_path):
    import cv2
    import numpy as np

    seen = []

    class Spy(FakeExtractor):
        def extract(self, rgb_frame, timestamp_ms):
            seen.append(np.array(rgb_frame))
            return super().extract(rgb_frame, timestamp_ms)

    video = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(10):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :32] = 255  # moitié gauche blanche
        writer.write(frame)
    writer.release()
    process_video(video, LABEL, lambda: Spy(), tmp_path / "a.csv", ExtractionOptions(0, 0, True))
    process_video(video, LABEL, lambda: Spy(), tmp_path / "b.csv", ExtractionOptions(0, 0, False))
    mirrored, plain = seen[0], seen[10]
    assert mirrored[:, 40].mean() > 200 and plain[:, 10].mean() > 200  # blanc à droite si miroir, à gauche sinon
    assert mirrored[:, 10].mean() < 50 and plain[:, 40].mean() < 50
