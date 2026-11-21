"""Ligne de commande : lit les arguments et lance le lot."""
import argparse
from pathlib import Path

from .batch_runner import EXIT_BAD_USAGE, BatchSettings, run_batch
from .config import DEFAULT_MODEL_PATH, DEFAULT_TRIM_S
from .extraction_options import ExtractionOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_landmarks",
        description="Extrait les landmarks de la main des vidéos de gestes (dossier <personne>/<geste>/<vidéo>).",
    )
    parser.add_argument("--input", required=True, type=Path, help="dossier local contenant les vidéos")
    parser.add_argument("--output", type=Path, default=Path("data/landmarks"), help="dossier de sortie (défaut : data/landmarks)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="modèle hand_landmarker.task")
    parser.add_argument("--trim-start", type=float, default=DEFAULT_TRIM_S, help="secondes retirées au début (défaut : 0.5)")
    parser.add_argument("--trim-end", type=float, default=DEFAULT_TRIM_S, help="secondes retirées à la fin (défaut : 0.5)")
    parser.add_argument("--no-mirror", action="store_true", help="désactive le miroir (actif par défaut, comme l'application)")
    parser.add_argument("--strict", action="store_true", help="ignore les vidéos dont le nom ne suit pas le format")
    parser.add_argument("--force", action="store_true", help="retraite les vidéos déjà traitées")
    parser.add_argument("--merge", action="store_true", help="fusionne tous les CSV en dataset.csv")
    parser.add_argument("--only", type=Path, help="ne traite que cette vidéo")
    parser.add_argument("--preview", action="store_true", help="écrit une vidéo avec les points dessinés (nécessite --only)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.preview and args.only is None:
        print("Erreur : --preview nécessite --only (aperçu d'une seule vidéo).")
        return EXIT_BAD_USAGE
    if not args.input.is_dir():
        print(f"Erreur : dossier d'entrée introuvable : {args.input}")
        return EXIT_BAD_USAGE
    settings = BatchSettings(
        input_dir=args.input, output_dir=args.output, model_path=args.model,
        options=ExtractionOptions(args.trim_start, args.trim_end, not args.no_mirror),
        strict=args.strict, force=args.force, merge=args.merge, only=args.only, preview=args.preview,
    )
    return run_batch(settings)
