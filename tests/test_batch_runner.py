import csv
from pathlib import Path

from extraction.batch_runner import BatchSettings, run_batch
from extraction.extraction_options import ExtractionOptions
from fake_extractor import factory
from video_builder import make_video

OPTIONS = ExtractionOptions(0.5, 0.5, True)


def _settings(tmp_path, **changes):
    values = dict(input_dir=tmp_path / "in", output_dir=tmp_path / "out", model_path=Path("inutile"), options=OPTIONS)
    values.update(changes)
    return BatchSettings(**values)


def _make(tmp_path, person, gesture, name):
    path = tmp_path / "in" / person / gesture / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return make_video(path)


def _report(tmp_path):
    with open(tmp_path / "out" / "report.csv", newline="", encoding="utf-8") as handle:
        return {row["video"]: row for row in csv.DictReader(handle)}


def test_lot_complet_chemin_sortie_et_rapport(tmp_path, capsys):
    _make(tmp_path, "p01", "clic_droit", "neon_incline_webcam.avi")
    _make(tmp_path, "p01", "drag", "IMG_0001.avi")
    _make(tmp_path, "p01", "salut", "jour_face_tel.avi")
    code = run_batch(_settings(tmp_path), factory())
    assert code == 0
    assert (tmp_path / "out" / "p01" / "clic_droit" / "neon_incline_webcam.csv").is_file()
    report = _report(tmp_path)
    assert report["p01/clic_droit/neon_incline_webcam.avi"]["status"] == "ok"
    assert report["p01/clic_droit/neon_incline_webcam.avi"]["frames_kept"] == "21"
    assert "non conforme" in report["p01/drag/IMG_0001.avi"]["warnings"]
    assert report["p01/salut/jour_face_tel.avi"]["status"] == "ignoree"
    assert "Résumé" in capsys.readouterr().out


def test_mode_strict_ignore_les_noms_non_conformes(tmp_path):
    _make(tmp_path, "p01", "drag", "IMG_0001.avi")
    run_batch(_settings(tmp_path, strict=True), factory())
    assert _report(tmp_path)["p01/drag/IMG_0001.avi"]["status"] == "ignoree"
    assert not (tmp_path / "out" / "p01" / "drag" / "IMG_0001.csv").exists()


def test_reprise_puis_force(tmp_path, capsys):
    _make(tmp_path, "p01", "drag", "jour_face_tel.avi")
    run_batch(_settings(tmp_path), factory())
    capsys.readouterr()
    calls = []
    run_batch(_settings(tmp_path), lambda: calls.append(1) or factory()())
    assert not calls and "déjà traitée" in capsys.readouterr().out
    run_batch(_settings(tmp_path, force=True), lambda: calls.append(1) or factory()())
    assert calls


def test_deux_executions_donnent_des_csv_identiques(tmp_path):
    _make(tmp_path, "p01", "drag", "jour_face_tel.avi")
    run_batch(_settings(tmp_path, output_dir=tmp_path / "out1"), factory(missing_every=4))
    run_batch(_settings(tmp_path, output_dir=tmp_path / "out2"), factory(missing_every=4))
    a = (tmp_path / "out1" / "p01" / "drag" / "jour_face_tel.csv").read_bytes()
    b = (tmp_path / "out2" / "p01" / "drag" / "jour_face_tel.csv").read_bytes()
    assert a == b


def test_only_et_merge_et_rapport_cumulatif(tmp_path):
    v1 = _make(tmp_path, "p01", "drag", "jour_face_tel.avi")
    v2 = _make(tmp_path, "p02", "scroll", "neon_face_webcam.avi")
    run_batch(_settings(tmp_path, only=v1), factory())
    assert not (tmp_path / "out" / "p02").exists()
    run_batch(_settings(tmp_path, only=v2, merge=True), factory())
    assert set(_report(tmp_path)) == {"p01/drag/jour_face_tel.avi", "p02/scroll/neon_face_webcam.avi"}
    with open(tmp_path / "out" / "dataset.csv", newline="", encoding="utf-8") as handle:
        assert len(list(csv.reader(handle))) == 1 + 21 * 2


def test_video_corrompue_ne_bloque_pas_le_lot(tmp_path):
    bad = tmp_path / "in" / "p01" / "drag" / "jour_face_tel.avi"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"pas une video")
    _make(tmp_path, "p01", "scroll", "jour_face_tel.avi")
    code = run_batch(_settings(tmp_path), factory())
    report = _report(tmp_path)
    assert code == 1
    assert report["p01/drag/jour_face_tel.avi"]["status"] == "erreur"
    assert report["p01/scroll/jour_face_tel.avi"]["status"] == "ok"


def test_preview_cree_une_video_annotee(tmp_path):
    video = _make(tmp_path, "p01", "drag", "jour_face_tel.avi")
    run_batch(_settings(tmp_path, only=video, preview=True), factory())
    preview = tmp_path / "out" / "preview" / "p01_drag_jour_face_tel.mp4"
    assert preview.is_file() and preview.stat().st_size > 0
