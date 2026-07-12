"""Portable path defaults for NegBioDB-CT runners."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
NEGBIODB_ROOT = Path(os.environ.get("NEGBIODB_ROOT", ROOT.parent / "Negative_result_DB")).expanduser()
CT_DB = Path(os.environ.get("NEGBIODB_CT_DB", NEGBIODB_ROOT / "data/negbiodb_ct.db")).expanduser()
CT_SPLITS = Path(os.environ.get("NEGBIODB_CT_SPLITS", NEGBIODB_ROOT / "exports/ct/negbiodb_ct_splits.csv")).expanduser()

