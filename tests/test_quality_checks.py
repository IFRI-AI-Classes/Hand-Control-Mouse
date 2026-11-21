from collections import Counter

from extraction.quality_checks import collect_warnings
from extraction.video_labels import VideoLabel
from extraction.video_stats import VideoStats

LABEL = VideoLabel("p01", "drag", "jour", "face", "webcam", "p01/drag/jour_face_webcam.mp4", True)


def _stats(**changes):
    base = dict(fps=30.0, width=640, height=480, frames_kept=100, frames_detected=100,
                side_counts=Counter({"Right": 100}))
    base.update(changes)
    return VideoStats(**base)


def test_video_saine_sans_avertissement():
    assert collect_warnings(_stats(), LABEL) == []


def test_detection_faible():
    assert any("détection" in w for w in collect_warnings(_stats(frames_detected=50, side_counts=Counter({"Right": 50})), LABEL))


def test_cote_incoherent():
    assert any("côté" in w for w in collect_warnings(_stats(side_counts=Counter({"Right": 60, "Left": 40})), LABEL))


def test_portrait_fps_bas_et_nom_non_conforme():
    bad = VideoLabel("p01", "drag", "inconnu", "inconnu", "inconnu", "x", False, ("nom de fichier non conforme : x",))
    warnings = collect_warnings(_stats(width=480, height=640, fps=15.0), bad)
    assert any("portrait" in w for w in warnings)
    assert any("FPS" in w for w in warnings)
    assert any("non conforme" in w for w in warnings)
