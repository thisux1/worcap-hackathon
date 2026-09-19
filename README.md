# WorCAP 2026 — previsão de precipitação mensal (América do Sul)

Solução para o desafio Kaggle `previsao-climatica-de-precipitacao-sobre-a-america-do-sul`
(INPE/WorCAP 2026): prever a precipitação média mensal (mm/dia) do mês seguinte
sobre grade 301×261 (lat −60..15, lon −90..−25).

## Resultado

| modelo | OOF (LOYO+embargo) | LB público (2023) |
|---|---|---|
| V0 climatologia mensal/pixel | 1.8081 | 1.85077 |
| **V0.5 ridge por pixel** | **1.7391** | **1.69966** (fit 1940–2022) |
| V1 LightGBM | ~1.85 subset | 1.84438 |
| V2 pixel-ridge (mini-Rodeo) | ~1.76 subset | 1.70886 |

Detalhes, fontes e decisões: `DECISIONS.md` (log de experimentos) e `ADR.md`
(arquitetura). Submissões em `submissions/`, validadas por `worcap/submission.py`.

## Modelo principal (V0.5)

Ridge multi-output por pixel sobre features mensais compartilhadas:

- **Índices oceânicos** (ERSSTv5): nino12/3/34/4, tna, tsa, atl3, tio, saod, amm,
  pmm_n, npac, ep_cp + lags (1–3m), médias móveis 3m, interações com janelas
  sazonais.
- **PDO real**: PC1 (EOF1) da anomalia de SST no Pacífico Norte (ERSSTv5,
  convenção Mantua/Deser — média global removida).
- **SAM-proxy**: MSLP zonal 40°S − 65°S padronizado (ERA5 monthly, CDS).
- **Médias regionais** de anomalia LOO das 9 variáveis atmosféricas oficiais
  em 5 macro-regiões.
- **NMME/SEAS5**: anomalias debiased de previsão dinâmica lead-1 (C3S
  `seasonal-monthly-single-levels`, NCEP-CFSv2 + ECMWF-SEAS5), regionais e
  per-pixel (V2). Climatologia do modelo exclui anos do fold (LOYO-safe).
- Alvo: anomalia padronizada por pixel (climatologia + σ mensais).

## Validação (sem vazamento temporal)

- LOYO: cada fold exclui ano de teste **±1 ano (embargo)** + holdout 2018–2022.
- Climatologia, padronização e anomalias NMME computadas **por fold**.
- Features externas restritas a ≤ fim do mês M−1 (mesma regra do teste).
- `worcap/folds.py`, `worcap/oof.py` (parquet por fold), `worcap/submission.py`
  (schema: ids = sample, sem NaN, ≥0, mm/dia).

## Dados externos utilizados

| fonte | uso | acesso |
|---|---|---|
| NOAA ERSSTv5 SST mensal (`data_ext/ersst_v5_sst_mnmean.nc`) | índices + PDO | público (NCEI) |
| ERA5 monthly MSLP −35..−70 (`data_ext/era5_mslp_southocean.nc`) | SAM-proxy | CDS API (conta gratuita) |
| C3S `seasonal-monthly-single-levels` NCEP+ECMWF, lead-1 (`data_ext/nmme/`) | features dinâmicas | CDS API + licenças |

Códigos-fonte dos dados e convenções em `worcap/indices.py` e `worcap/nmme.py`.

## Reprodução

```bash
conda env create -f environment.yml && conda activate worcap
# dados oficiais na raiz (13 .nc) + data_ext/ (ERSST, ERA5-SLP, NMME)
python -m worcap.indices                 # índices ERSST + SAM + PDO
python -c "from worcap.features import build_shared_table; build_shared_table()"
python scripts/v0_climatologia.py        # baseline V0
python scripts/v05_ridge.py              # OOF do ridge (78 folds)
python scripts/v05_ridge.py --submit --all   # gera submissions/v05_ridge_all.csv
```

Execução pesada (GBM, pixel-ridge, evals) roda em kernel Kaggle privado
(`kaggle_kernel/`), usando o dataset privado `worcap-assets2` — ver `ADR.md`.

## Variáveis de ambiente

`WORCAP_DATA` (dir dos .nc oficiais), `WORCAP_EXT` (data_ext),
`WORCAP_SUB`, `WORCAP_OOF`, `WORCAP_CACHE` (gravável), `WORCAP_EAGER=1`
(carrega variáveis em RAM — usar no Kaggle).
