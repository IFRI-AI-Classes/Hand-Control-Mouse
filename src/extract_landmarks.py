"""Point d'entrée : python src/extract_landmarks.py --input <dossier_videos> [options]"""
import sys

from extraction.cli import main

if __name__ == "__main__":
    sys.exit(main())
