from extraction.timeline import build_keep_window, frame_timestamp_ms


def test_timestamp_depuis_le_fps():
    assert frame_timestamp_ms(0, 30) == 0
    assert frame_timestamp_ms(30, 30) == 1000
    assert frame_timestamp_ms(1, 10) == 100


def test_fenetre_avec_trim():
    window = build_keep_window(frame_count=30, fps=10, trim_start_s=0.5, trim_end_s=0.5)
    assert (window.start_ms, window.end_ms) == (500, 2500)
    assert window.contains(500) and window.contains(2500)
    assert not window.contains(400) and not window.contains(2600)


def test_duree_inconnue_pas_de_borne_de_fin():
    window = build_keep_window(frame_count=0, fps=30, trim_start_s=0.5, trim_end_s=0.5)
    assert window.end_ms is None and window.contains(10_000_000)
