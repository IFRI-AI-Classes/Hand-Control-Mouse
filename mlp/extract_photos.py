"""Extrait les landmarks des photos (.jpg, .png) rangées comme les vidéos : <personne>/<geste>/<photo>.

Même format de CSV, mêmes réglages MediaPipe et même miroir que l'extraction des vidéos.
Lancer depuis la racine du dépôt :  python -m mlp.extract_photos
"""
import sys
from pathlib import Path

import cv2

from .config import PHOTOS_CSV, PHOTOS_DIR, ROOT

sys.path.insert(0, str(ROOT / "src"))
from extraction.config import DEFAULT_MODEL_PATH, GESTURES, MIRROR_DEFAULT  # noqa: E402
from extraction.csv_io import AtomicCsvWriter  # noqa: E402
from extraction.csv_schema import COLUMNS, build_row  # noqa: E402
from extraction.data_models import FrameRecord  # noqa: E402
from extraction.hand_extractor import create_mediapipe_extractor  # noqa: E402
from extraction.video_labels import VideoLabel  # noqa: E402

PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    photos = sorted(p for p in Path(PHOTOS_DIR).rglob("*") if p.suffix.lower() in PHOTO_EXTENSIONS)
    found = 0
    with AtomicCsvWriter(PHOTOS_CSV, COLUMNS) as writer:
        for photo in photos:
            rel = photo.relative_to(PHOTOS_DIR)
            if len(rel.parts) != 3 or rel.parts[1].lower() not in GESTURES:
                continue                                   # gestes.png, README, etc.
            person, gesture = rel.parts[0], rel.parts[1].lower()
            image = cv2.imread(str(photo))
            if image is None:
                print(f"  ! illisible : {rel.as_posix()}")
                continue
            if MIRROR_DEFAULT:
                image = cv2.flip(image, 1)
            extractor = create_mediapipe_extractor(DEFAULT_MODEL_PATH)   # un détecteur par photo
            observation = extractor.extract(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0)
            extractor.close()
            label = VideoLabel(person, gesture, "inconnu", "inconnu", "photo", rel.as_posix(), True)
            record = FrameRecord(0, 0, image.shape[1], image.shape[0])
            writer.write_row(build_row(label, record, observation))
            found += observation is not None
            print(f"{rel.as_posix()} : {'main OK' if observation else 'PAS de main'}")
    print(f"{found}/{len(photos)} photos avec une main -> {PHOTOS_CSV}")


if __name__ == "__main__":
    main()
