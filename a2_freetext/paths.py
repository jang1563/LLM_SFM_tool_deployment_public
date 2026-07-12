"""Portable path defaults for A2 free-text experiments."""

from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
A2_DIR = Path(os.environ.get("A2_FREETEXT_DIR", Path(__file__).resolve().parent)).expanduser().resolve()
NEGBIODB_ROOT = Path(os.environ.get("NEGBIODB_ROOT", REPO_ROOT.parent / "Negative_result_DB")).expanduser()
CT_DB = Path(os.environ.get("NEGBIODB_CT_DB", NEGBIODB_ROOT / "data/negbiodb_ct.db")).expanduser()
CHEMBL_DB = Path(
    os.environ.get(
        "CHEMBL_DB",
        Path.home() / "chembl_a2/chembl_37/chembl_37_sqlite/chembl_37.db",
    )
).expanduser()

