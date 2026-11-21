import pytest

from extraction.csv_schema import COLUMNS, LANDMARK_COLUMNS, META_COLUMNS, build_row
from extraction.data_models import FrameRecord, HandObservation
from extraction.video_labels import VideoLabel

LABEL = VideoLabel("p01", "drag", "jour", "face", "webcam", "p01/drag/jour_face_webcam.mp4", True)
RECORD = FrameRecord(3, 100, 64, 48)


def test_76_colonnes_dans_le_bon_ordre():
    assert len(COLUMNS) == 76 and len(META_COLUMNS) == 13 and len(LANDMARK_COLUMNS) == 63
    assert COLUMNS[:3] == ["person", "gesture", "lighting"]
    assert COLUMNS[13:16] == ["lm0_x", "lm0_y", "lm0_z"] and COLUMNS[-1] == "lm20_z"


def test_ligne_avec_main():
    observation = HandObservation(0.91234, "Left", tuple((0.5, 0.25, -0.1) for _ in range(21)))
    row = build_row(LABEL, RECORD, observation)
    assert len(row) == 76
    assert row[10:13] == [1, "0.9123", "Left"] and row[13:16] == ["0.500000", "0.250000", "-0.100000"]


def test_ligne_sans_main():
    row = build_row(LABEL, RECORD, None)
    assert len(row) == 76 and row[10] == 0 and set(row[11:]) == {""}


def test_nombre_de_landmarks_incorrect():
    with pytest.raises(ValueError):
        build_row(LABEL, RECORD, HandObservation(0.9, "Left", ((0, 0, 0),)))
