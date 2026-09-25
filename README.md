<div align="center">
  <a href="#root"><img src="./assets/banner.svg?v=1" alt="WORCAP 2026" width="100%"/></a>
</div>

<table width="100%">
  <tr>
    <td width="50%" valign="top">
      <pre lang="bash"><code>$ worcap / briefing
------------------------------------------------
• Desafio : precipitação média mensal (mm/dia)
• Grade   : 301×261 — lat −60..15, lon −90..−25
• Métrica : RMSE absoluto · público 2023 / privado 2024
• Treino  : ERA5 1940–2022 · features atmosféricas T−1</code></pre>
    </td>
    <td width="50%" valign="top">
      <pre lang="python"><code>class Solucao:
    stack    = ["ridge λ=1000", "pixel-ridge λ2=300",
                "blend 65/35", "enso_offset β"]
    validacao = "LOYO + embargo ±1a"
    finals    = ["splice_A_b33_24",
                 "blend_ridge_v2_65"]
    melhor_lb = 1.68586</code></pre>
    </td>
  </tr>
</table>

### ❯ badges

<p align="left">
  <img src="https://img.shields.io/badge/Kaggle-0D1117?style=flat-square&logo=kaggle&logoColor=ff2a5f&labelColor=0D1117&color=ff2a5f" alt="Kaggle" />
  <img src="https://img.shields.io/badge/Python-0D1117?style=flat-square&logo=python&logoColor=ff2a5f&labelColor=0D1117&color=ff2a5f" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-0D1117?style=flat-square&logo=scikitlearn&logoColor=ff2a5f&labelColor=0D1117&color=ff2a5f" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/xarray-0D1117?style=flat-square&logoColor=ff2a5f&labelColor=0D1117&color=ff2a5f" alt="xarray" />
  <img src="https://img.shields.io/badge/License-MIT-0D1117?style=flat-square&logoColor=39ff14&labelColor=0D1117&color=39ff14" alt="MIT" />
  <img src="https://img.shields.io/badge/LB_público-1.68586-0D1117?style=flat-square&labelColor=0D1117&color=f1fa8c" alt="LB 1.68586" />
  <img src="https://img.shields.io/badge/status-concluído-0D1117?style=flat-square&labelColor=0D1117&color=6272a4" alt="status" />
</p>

---

### ❯ pipeline

<div align="center">
  <img src="./assets/pipeline.svg?v=1" alt="Arquitetura" width="100%"/>
</div>

Features mensais compartilhadas: índices oceânicos do ERSSTv5 (nino12/3/34/4, tna,
tsa, atl3, tio, saod, amm, pmm_n, npac, ep_cp + lags e médias móveis), PDO como
PC1 do Pacífico Norte, proxy de SAM do ERA5-MSLP, médias regionais leave-one-out
das 9 variáveis oficiais e NMME/SEAS5 lead-1 debiased. O alvo é a anomalia
padronizada por pixel. O `enso_offset` soma `β·(clim_fase − clim_full)` só nos
meses-alvo cuja fase ENSO na origem T−1 sai da faixa neutra (n34 ≥ 0.8 / ≤ −0.8).

---

### ❯ resultados

<div align="center">
  <img src="./assets/results.svg?v=1" alt="Scores no leaderboard público" width="100%"/>
</div>

| modelo | OOF (LOYO+embargo) | LB público (2023) |
|---|---|---|
| V0 climatologia mensal/pixel | 1.8081 | 1.85077 |
| V0.5 ridge por pixel | 1.7391 | 1.69966 (fit 1940–2022) |
| V1 LightGBM | ~1.85 subset | 1.84438 |
| V2 pixel-ridge (mini-Rodeo) | ~1.76 subset | 1.70886 |
| blend 65% V0.5 + 35% V2 | ~1.75 | 1.69816 |
| blend + offset ENSO (β=0.25, thr=0.8) | — | 1.68586 |

---

### ❯ seleção_final

O privado é 2024, um ano-decay de El Niño, e o Kaggle fica com a melhor das duas
finais marcadas (min-of-2). A escolha foi um barbell: upside na dose ENSO, piso
no blend sem offset.

| slot | arquivo | papel |
|---|---|---|
| final1 | `splice_A_b33_24` | 2023 = A; 2024 com β=0.33 (ótimo OOF dos anos-decay) |
| final2 | `blend_ridge_v2_65` | sem offset — cobre o modo de falha visto em 2016 |

Raciocínio completo, divergências e vereditos: `FINAL_OPTIONS.md` e
`DECISIONS.md` (sessões 10–12).

---

### ❯ anti_leak

- LOYO com embargo ±1 ano; holdout interno 2018–2022 só para seleção.
- Climatologia, σ, compósitos ENSO e debias NMME recalculados dentro de cada fold.
- Toda feature usa apenas dados até o fim do mês M−1 (mesma regra do teste).
- `worcap/folds.py`, `worcap/oof.py`, `worcap/submission.py`; testes em `tests/`.

---

### ❯ dados_externos

| fonte | uso | acesso |
|---|---|---|
| NOAA ERSSTv5 SST mensal (`data_ext/ersst_v5_sst_mnmean.nc`) | índices + PDO | público (NCEI) |
| ERA5 monthly MSLP −35..−70 (`data_ext/era5_mslp_southocean.nc`) | proxy SAM | CDS API (conta gratuita) |
| C3S `seasonal-monthly-single-levels` NCEP+ECMWF, lead-1 (`data_ext/nmme/`) | features dinâmicas | CDS API + licenças |

Proveniência, hashes e licenças: `DATA_MANIFESTO.md`.

---

### ❯ reprodução

```bash
conda env create -f environment.yml && conda activate worcap
export WORCAP_DATA=. WORCAP_EXT=data_ext WORCAP_SUB=submissions
python -m worcap.indices                 # índices ERSST + SAM + PDO
python -c "from worcap.features import build_shared_table; build_shared_table()"
python scripts/v0_climatologia.py        # baseline
python scripts/v05_ridge.py --submit --all
```

O passo a passo das duas finais está em `REPRODUCE.md`. A execução pesada rodou
em kernel Kaggle (`kaggle_kernel/`, dataset `worcap-assets2`) — ver `ADR.md`.

Variáveis: `WORCAP_DATA`, `WORCAP_EXT`, `WORCAP_SUB`, `WORCAP_OOF`,
`WORCAP_CACHE`, `WORCAP_EAGER=1` (carrega em RAM, usar no Kaggle).

---

### ❯ estrutura

<pre lang="text"><code>worcap/          núcleo — indices, features, folds, oof, pixel_ridge, enso, nmme, submission
scripts/         CLI dos experimentos — v0_climatologia, v05_ridge, v2_pixel_ridge,
                 blend, enso_offset, splice_year, gamma_warm, damp, report
kaggle_*/        kernels e assets usados na execução remota
tests/           causalidade e schema (test_invariance, test_enso, ...)
submissions/     CSVs gerados (validados por worcap/submission.py)
data_ext/        dados externos (ERSSTv5, ERA5-MSLP, NMME) — ver manifesto</code></pre>

---

### ❯ docs

`ADR.md` · arquitetura — `DECISIONS.md` · log de experimentos —
`FINAL_OPTIONS.md` · decisão das finais — `DATA_MANIFESTO.md` · dados externos —
`RULES.md` · regras — `REPRODUCE.md` · reprodução
