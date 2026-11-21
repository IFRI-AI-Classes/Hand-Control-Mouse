import pytest

from extraction.video_labels import LabelError, parse_video_label


def _video(tmp_path, *parts):
    path = tmp_path.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_chemin_valide(tmp_path):
    label = parse_video_label(_video(tmp_path, "p01", "clic_droit", "neon_incline_webcam.mp4"), tmp_path)
    assert (label.person, label.gesture) == ("p01", "clic_droit")
    assert (label.lighting, label.orientation, label.source) == ("neon", "incline", "webcam")
    assert label.filename_ok and label.video == "p01/clic_droit/neon_incline_webcam.mp4"


def test_casse_ignoree(tmp_path):
    label = parse_video_label(_video(tmp_path, "p01", "Drag", "Jour_Face_TEL.mp4"), tmp_path)
    assert (label.gesture, label.lighting, label.orientation, label.source) == ("drag", "jour", "face", "tel")


def test_dossier_de_geste_inconnu(tmp_path):
    with pytest.raises(LabelError):
        parse_video_label(_video(tmp_path, "p01", "salut", "jour_face_tel.mp4"), tmp_path)


def test_structure_incorrecte(tmp_path):
    with pytest.raises(LabelError):
        parse_video_label(_video(tmp_path, "clic_droit", "jour_face_tel.mp4"), tmp_path)


def test_nom_de_fichier_non_conforme_donne_inconnu(tmp_path):
    label = parse_video_label(_video(tmp_path, "p01", "scroll", "IMG_0042.mp4"), tmp_path)
    assert (label.lighting, label.orientation, label.source) == ("inconnu",) * 3
    assert not label.filename_ok and label.problems


def test_video_hors_du_dossier_d_entree(tmp_path):
    ailleurs = tmp_path / "ailleurs"
    entree = tmp_path / "entree"
    entree.mkdir()
    with pytest.raises(LabelError):
        parse_video_label(_video(ailleurs, "p01", "drag", "jour_face_tel.mp4"), entree)
