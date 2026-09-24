# WorCAP 2026 — previsão de precipitação mensal (América do Sul)

Solução para o desafio Kaggle `previsao-climatica-de-precipitacao-sobre-a-america-do-sul`
(INPE/WorCAP 2026): prever a precipitação média mensal (mm/dia) do mês seguinte
sobre grade 301×261 (lat −60..15, lon −90..−25).

## Resultado

| modelo | OOF (LOYO+embargo) | LB público (2023) |
|---|---|---|
| V0 climatologia mensal/pixel | 1.8081 | 1.85077 |
| V0.5 ridge por pixel | 1.7391 | 1.69966 (fit 1940–2022) |
| V1 LightGBM | ~1.85 subset | 1.84438 |
| V2 pixel-ridge (mini-Rodeo) | ~1.76 subset | 1.70886 |
| blend 65% V0.5 + 35% V2 | ~1.75 | 1.69816 |
| blend + offset ENSO (β=0.25, thr=0.8) | — | 1.68586 |

Finais escolhidas (privado = 2024): `splice_A_b33_24` (lado-2024 com β=0.33,
dose sugerida pelos anos-decay do OOF) e `blend_ridge_v2_65` (blend sem offset,
para cobrir o caso em que o offset não transfere, como aconteceu em 2016).
O raciocínio completo está em `FINAL_OPTIONS.md` e em `DECISIONS.md` sessão 12.

Detalhes de dados e decisões: `DECISIONS.md` (log de experimentos), `ADR.md`
(arquitetura) e `REPRODUCE.md` (como gerar os dois CSVs finais). Submissões em
`submissions/`, validadas por `worcap/submission.py`.

## Modelo principal (V0.5 + V2)

Ridge multi-output por pixel sobre features mensais compartilhadas:

- Índices oceânicos (ERSSTv5): nino12/3/34/4, tna, tsa, atl3, tio, saod, amm,
  pmm_n, npac, ep_cp, mais lags de 1–3 meses, médias móveis de 3m e interações
  sazonais.
- PDO calculado como PC1 (EOF1) da anomalia de SST no Pacífico Norte (ERSSTv5,
  convenção Mantua/Deser, média global removida).
- Proxy de SAM: MSLP zonal 40°S − 65°S padronizado (ERA5 monthly, CDS).
- Médias regionais de anomalia leave-one-out das 9 variáveis atmosféricas
  oficiais em 5 macro-regiões.
- NMME/SEAS5: anomalias debiased de previsão dinâmica lead-1 (C3S
  `seasonal-monthly-single-levels`, NCEP-CFSv2 + ECMWF-SEAS5), regionais e
  por pixel (V2). A climatologia do modelo exclui os anos do fold.
- Alvo: anomalia padronizada por pixel (climatologia + σ mensais).

O V2 é um segundo ridge por pixel no resíduo do V0.5, com features locais
(lags, vizinhança). O blend final junta os dois em espaço de precipitação.

## Validação (sem vazamento temporal)

- LOYO: cada fold exclui o ano de teste e os vizinhos ±1 (embargo), com holdout
  interno 2018–2022.
- Climatologia, padronização, compósitos ENSO e anomalias NMME computados
  dentro de cada fold.
- Features externas restritas a dados até o fim do mês M−1 (mesma regra do teste).
- `worcap/folds.py`, `worcap/oof.py` (parquet por fold), `worcap/submission.py`
  (schema: ids iguais ao sample, sem NaN, ≥0, mm/dia).

## Dados externos utilizados

| fonte | uso | acesso |
|---|---|---|
| NOAA ERSSTv5 SST mensal (`data_ext/ersst_v5_sst_mnmean.nc`) | índices + PDO | público (NCEI) |
| ERA5 monthly MSLP −35..−70 (`data_ext/era5_mslp_southocean.nc`) | proxy SAM | CDS API (conta gratuita) |
| C3S `seasonal-monthly-single-levels` NCEP+ECMWF, lead-1 (`data_ext/nmme/`) | features dinâmicas | CDS API + licenças |

Convenções e proveniência completa: `worcap/indices.py`, `worcap/nmme.py` e
`DATA_MANIFESTO.md`.

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

A geração das duas finais passo a passo está em `REPRODUCE.md`. Execução pesada
(GBM, pixel-ridge, evals) rodou em kernel Kaggle privado (`kaggle_kernel/`),
usando o dataset privado `worcap-assets2`; ver `ADR.md`.

## Variáveis de ambiente

`WORCAP_DATA` (dir dos .nc oficiais), `WORCAP_EXT` (data_ext),
`WORCAP_SUB`, `WORCAP_OOF`, `WORCAP_CACHE` (gravável), `WORCAP_EAGER=1`
(carrega variáveis em RAM — usar no Kaggle).
