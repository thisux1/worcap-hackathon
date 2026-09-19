"""Blend de submissoes em espaco de y (media ponderada de mm/dia).

Uso: python scripts/blend.py out.csv w1 a.csv w2 b.csv [w3 c.csv ...]
Pesos normalizados; ids conferidos contra o sample antes de gravar.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from worcap.submission import load_sample_index, validate_submission


def main():
    out = sys.argv[1]
    pairs = sys.argv[2:]
    assert len(pairs) % 2 == 0
    ws = [float(pairs[i]) for i in range(0, len(pairs), 2)]
    fs = [pairs[i + 1] for i in range(0, len(pairs), 2)]
    ws = np.array(ws) / np.sum(ws)

    ids = load_sample_index()["id"]
    acc = None
    for w, f in zip(ws, fs):
        df = pd.read_csv(f)
        assert df["id"].equals(ids), f"ids divergem em {f}"
        v = df["tp_mm_day"].values
        acc = w * v if acc is None else acc + w * v
    sub = pd.DataFrame({"id": ids, "tp_mm_day": np.clip(acc, 0, None)})
    sub.to_csv(out, index=False, float_format="%.6f")
    print(out, validate_submission(out))


if __name__ == "__main__":
    main()
