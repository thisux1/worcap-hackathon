"""Report visual do estado da competicao — gera report.png + resumo em texto.

Painel A: submissoes no tempo (LB publico) com referencias.
Painel B: leaderboard (top-20, nossa posicao marcada).
Painel C: OOF vs LB por modelo (gap de transferencia p/ regime 2023).
Painel D: curva de damping OOF vs resultado LB (divergencia regime).
Painel E: peso do blend no LB.
Painel F: status e proximos passos.

Uso: .venv/bin/python scripts/report.py [--fetch]   (--fetch puxa LB/submits ao vivo)
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "report.png"

CLIM = 1.85077          # baseline oficial (climatologia)
BEST = 1.69816          # melhor LB atual

# (data UTC, nome, LB publico) — dump da API 19/09
SUBS = [
    ("17/09", "v0_clim",              1.85077),
    ("17/09", "v05_ridge",            1.72679),
    ("17/09", "v1_lgbm",              1.84438),
    ("17/09", "v05_ridge_all",        1.69966),
    ("17/09", "v2_pixel_ridge",       1.70886),
    ("18/09", "v05_sam_pdo_nmme",     1.73220),
    ("18/09", "blend_ridge_v2(60/40)",1.69836),
    ("18/09", "v2_nmme",              1.75175),
    ("18/09", "blend_ridge_v2_50",    1.69908),
    ("18/09", "blend_sp20",           1.70403),
    ("19/09", "blend_v2_d80",         1.70584),
    ("19/09", "blend_v2_65_d80",      1.70599),
    ("19/09", "blend_ridge_v2_65",    1.69816),
    ("19/09", "v05_ridge_all_d80",    1.70885),
    ("19/09", "blend_v2_d70",         1.71410),
]

LB = [  # leaderboard publico 18/09 ~23h UTC (top-8 + contexto)
    ("Rain-NP-Hard",    1.49924),
    ("RainForcement",   1.57591),
    ("NoTime",          1.68186),
    ("MatHab",          1.68237),
    ("Thiago (nos)",    1.69836),
    ("davis denner",    1.71456),
    ("Bruno Simoes",    1.71488),
    ("RainPredict",     1.75168),
]

# (modelo, OOF/CV, LB) — OOF onde existe
OOF_LB = [
    ("clim",        1.808,  1.85077),
    ("ridge 40-17", 1.7379, 1.72679),
    ("ridge_all",   1.7391, 1.69966),
    ("v2 (7a)",     1.7596, 1.70886),
    ("lgbm",        None,   1.84438),
]

DAMP_OOF = [(0.50, 1.7470), (0.60, 1.7433), (0.70, 1.7412), (0.75, 1.7407),
            (0.80, 1.7406), (0.85, 1.7409), (0.90, 1.7416), (0.95, 1.7427),
            (1.00, 1.7442)]
DAMP_LB = [("blend 60/40", 1.00, 1.69836), ("blend 60/40", 0.80, 1.70584),
           ("blend 60/40", 0.70, 1.71410), ("ridge_all", 1.00, 1.69966),
           ("ridge_all", 0.80, 1.70885), ("blend 65/35", 0.80, 1.70599)]

BLEND_W = [(0.5, 1.69908), (0.6, 1.69836), (0.65, 1.69816), (0.8, 1.70403),
           (1.0, 1.69966)]


def fetch_lb():
    """Leaderboard ao vivo via kaggle CLI."""
    try:
        out = subprocess.run(
            [".venv/bin/kaggle", "competitions", "leaderboard",
             "previsao-climatica-de-precipitacao-sobre-a-america-do-sul",
             "--show"], capture_output=True, text=True, timeout=60).stdout
        rows = []
        for ln in out.splitlines():
            p = ln.split()
            if len(p) >= 4 and p[0].isdigit():
                rows.append((" ".join(p[1:-2]), float(p[-1])))
        return rows[:20] or LB
    except Exception:
        return LB


def main():
    lb = fetch_lb() if "--fetch" in sys.argv else LB

    fig = plt.figure(figsize=(19, 11), facecolor="#0e1117")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.06, right=0.97, top=0.90, bottom=0.08)
    fig.suptitle("WorCAP 2026 — estado da solucao (19/09 ~01:00 UTC)\n"
                 "previsao de precipitacao mensal | metrica RMSE mm/dia | "
                 "publico=2023, privado=2024", color="w", fontsize=14, y=0.97)

    def style(ax, title):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.set_title(title, color="#e6edf3", fontsize=10, loc="left", pad=8)
        ax.grid(alpha=0.15, color="#8b949e")

    # A — submissoes no tempo
    ax = fig.add_subplot(gs[0, 0])
    style(ax, "A · Submissoes no tempo (LB publico)")
    xs = np.arange(len(SUBS))
    cols = ["#ff7b72" if s[2] > BEST + 0.02 else "#79c0ff" if s[2] > BEST
            else "#56d364" for s in SUBS]
    ax.scatter(xs, [s[2] for s in SUBS], c=cols, s=42, zorder=3)
    ax.axhline(CLIM, color="#d29922", ls="--", lw=1)
    ax.text(len(SUBS) - 1, CLIM + 0.004, "climatologia 1.851", color="#d29922",
            fontsize=8, ha="right")
    ax.axhline(BEST, color="#56d364", ls=":", lw=1)
    ax.text(len(SUBS) - 1, BEST - 0.014, "melhor 1.69816", color="#56d364",
            fontsize=8, ha="right")
    ticks = [i for i in range(len(SUBS))
             if i == 0 or SUBS[i][0] != SUBS[i - 1][0]]
    for t in ticks:
        ax.axvline(t - 0.5, color="#30363d", lw=0.6)
        ax.text(t, 1.872, SUBS[t][0], color="#8b949e", fontsize=8, ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[1] for s in SUBS], rotation=75, fontsize=6.5)
    ax.set_ylim(1.66, 1.88)
    ax.set_ylabel("RMSE", color="#c9d1d9", fontsize=9)

    # B — leaderboard
    ax = fig.add_subplot(gs[0, 1])
    style(ax, "B · Leaderboard publico (suspeitos de leak marcados)")
    names = [t[0] for t in lb][::-1]
    vals = [t[1] for t in lb][::-1]
    cols = []
    for n, v in zip(names, vals):
        if "Thiago" in n:
            cols.append("#56d364")
        elif v < 1.62:
            cols.append("#ff7b72")
        else:
            cols.append("#79c0ff")
    ax.barh(names, vals, color=cols, height=0.65)
    ax.axvline(BEST, color="#56d364", ls=":", lw=1)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.5f}", color="#c9d1d9", fontsize=7, va="center")
    ax.set_xlim(min(vals) - 0.03, max(vals) + 0.10)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.text(max(vals) + 0.005, 0.4, "vermelho = possivel leak\n"
            "(revisao oficial anunciada)", color="#ff7b72", fontsize=7.5,
            va="bottom")

    # C — OOF vs LB
    ax = fig.add_subplot(gs[0, 2])
    style(ax, "C · CV/OOF vs LB — gap de transferencia p/ 2023")
    names = [m[0] for m in OOF_LB]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, [m[1] if m[1] else 0 for m in OOF_LB], w,
           color="#79c0ff", label="CV/OOF")
    ax.bar(x + w / 2, [m[2] for m in OOF_LB], w, color="#f0a868", label="LB")
    ax.axhline(CLIM, color="#d29922", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, fontsize=8)
    for i, m in enumerate(OOF_LB):
        if m[1]:
            ax.text(i, max(m[1], m[2]) + 0.008, f"{m[2]-m[1]:+.3f}",
                    ha="center", color="#c9d1d9", fontsize=8)
    ax.legend(fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9")
    ax.set_ylim(1.68, 1.90)
    ax.set_ylabel("RMSE", color="#c9d1d9", fontsize=9)

    # D — damping: OOF vs LB
    ax = fig.add_subplot(gs[1, 0])
    style(ax, "D · Damping: OOF diz bom (−0.004), LB diz mau (+0.008)")
    ax.plot([d[0] for d in DAMP_OOF], [d[1] for d in DAMP_OOF],
            "o-", color="#79c0ff", label="OOF ridge")
    ax.set_xlabel("gamma (encolhimento da anomalia)", color="#c9d1d9",
                  fontsize=8)
    ax.set_ylabel("RMSE OOF", color="#79c0ff", fontsize=9)
    ax2 = ax.twinx()
    seen = set()
    for nome, g, v in DAMP_LB:
        lbl = "LB" if "LB" not in seen else None
        seen.add("LB")
        ax2.scatter(g, v, color="#ff7b72", s=55, marker="s", zorder=3, label=lbl)
    ax2.set_ylabel("RMSE LB", color="#ff7b72", fontsize=9)
    ax2.tick_params(colors="#ff7b72", labelsize=8)
    ax2.set_ylim(1.69, 1.72)
    ax.legend(fontsize=8, loc="lower right", facecolor="#161b22",
              labelcolor="#c9d1d9")
    ax2.legend(fontsize=8, loc="upper left", facecolor="#161b22",
               labelcolor="#c9d1d9")
    ax.set_title("D · Damping: OOF diz bom, LB diz mau", color="#e6edf3",
                 fontsize=10, loc="left", pad=8)

    # E — peso do blend no LB
    ax = fig.add_subplot(gs[1, 1])
    style(ax, "E · Peso do ridge no blend (LB)")
    ax.plot([b[0] for b in BLEND_W], [b[1] for b in BLEND_W], "o-",
            color="#f0a868")
    for w_, v in BLEND_W:
        ax.annotate(f"{v:.5f}", (w_, v), textcoords="offset points",
                    xytext=(0, 7), color="#c9d1d9", fontsize=7.5, ha="center")
    ax.axhline(1.70886, color="#8b949e", ls=":", lw=1)
    ax.text(0.42, 1.7093, "v2 puro", color="#8b949e", fontsize=7.5)
    ax.set_xlabel("peso do ridge (1-w = v2)", color="#c9d1d9", fontsize=9)
    ax.set_ylabel("RMSE", color="#c9d1d9", fontsize=9)
    ax.set_ylim(1.694, 1.715)

    # F — status
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor("#161b22")
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("F · Status e proximos passos", color="#e6edf3",
                 fontsize=10, loc="left", pad=8)
    txt = (
        "LB publico: 5o lugar — 1.69816 (blend 65/35)\n"
        "baseline climatologia: 1.85077  (−8.2%)\n"
        "dias restantes: ~4 (fim 23/09)\n"
        "\n"
        "REJEITADO empiricamente:\n"
        "  damping g=0.80 (+0.008 LB p/ -0.004 OOF)\n"
        "  features SAM/PDO/NMME (+0.03 LB)\n"
        "  LGBM (1.844)\n"
        "\n"
        "EM CURSO (kernel v22):\n"
        "  grid (peso x gamma) x regime gap1/gap2\n"
        "  avaliacao de recursao autorregressiva\n"
        "\n"
        "FINAIS PROVAVEIS:\n"
        "  1: blend_ridge_v2_65 (melhor LB)\n"
        "  2: v05_ridge_all (hedge, sem damping)\n"
        "  + splice 2023/2024 se grid gap-2 justificar\n"
        "\n"
        "AUDITORIA: limpa (test_invariance 4/4)\n"
        "top-2 do LB suspeitos de leak → revisao oficial")
    ax.text(0.02, 0.97, txt, color="#c9d1d9", fontsize=8.6, va="top",
            family="monospace")

    fig.savefig(OUT, dpi=130, facecolor=fig.get_facecolor())
    print("salvo:", OUT)

    print("\n=== RESUMO ===")
    print(f"melhor LB: {BEST} (blend_ridge_v2_65) | baseline: {CLIM}")
    print("submits hoje: 5/5 usados | proximos: 00:00 UTC")
    print("kernel v22: grid(w,g)xgap + recursao — decide finais")


if __name__ == "__main__":
    main()
