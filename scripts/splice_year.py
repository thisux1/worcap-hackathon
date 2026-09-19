"""Splice por ano-alvo: linhas 2023_* de A + linhas 2024_* de B.

Publico (2023) mede A; privado (2024) mede B — permite otimizar a metade
privada (regime lag 13-24) sem custo no placar publico, e manter ambas as
finais no top-2 publico (auto-selecao do Kaggle como fallback).

Uso: python scripts/splice_year.py A_publico.csv B_privado.csv out.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worcap.submission import validate_submission


def splice(a_pub: str, b_priv: str, out: str) -> pd.DataFrame:
    a = pd.read_csv(a_pub).set_index("id")
    b = pd.read_csv(b_priv).set_index("id")
    assert a.index.equals(b.index), "ids divergentes"
    mask24 = a.index.str.startswith("2024_")
    res = a.copy()
    res.loc[mask24, "tp_mm_day"] = b.loc[mask24, "tp_mm_day"]
    res = res.reset_index()
    res.to_csv(out, index=False)
    print(out, validate_submission(out))
    return res


if __name__ == "__main__":
    splice(sys.argv[1], sys.argv[2], sys.argv[3])
