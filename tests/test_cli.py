from extraction.cli import main


def test_preview_sans_only_refuse(tmp_path):
    assert main(["--input", str(tmp_path), "--preview"]) == 2


def test_dossier_d_entree_introuvable(tmp_path):
    assert main(["--input", str(tmp_path / "absent")]) == 2


def test_modele_introuvable(tmp_path):
    assert main(["--input", str(tmp_path), "--model", str(tmp_path / "absent.task")]) == 2
