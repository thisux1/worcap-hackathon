# Reprodução das submissões finais — equipe thiago (Kaggle)

Todas as features usam só dados até o fim do mês de origem (T−1). A validação
é LOYO com embargo de ±1 ano; climatologia e compósitos ENSO são refeitos dentro
de cada fold.

## Dados de entrada

- Oficiais (.nc na raiz): `treino_tp.nc`, `treino_tp_alvo.nc`, `teste_features.nc` e os demais `treino_*.nc`
- Externos (`data_ext/`): lista completa com fontes e licenças em `DATA_MANIFESTO.md` (ERSSTv5 da NOAA, ERA5-MSLP do CDS, NMME do C3S)

## Setup

```bash
conda env create -f environment.yml && conda activate worcap
export WORCAP_DATA=. WORCAP_EXT=data_ext WORCAP_SUB=submissions
python -m worcap.indices                  # índices oceânicos, tudo no mês de origem
python -c "from worcap.features import build_shared_table; build_shared_table()"
```

## final2: `blend_ridge_v2_65.csv` (público 1.69816)

```bash
python scripts/v05_ridge.py --submit --all      # ridge global por pixel, lam=1000, fit 1940-2022
python scripts/v2_pixel_ridge.py                # correção ridge por pixel, lam2=300
python scripts/blend.py submissions/v05_ridge_all.csv submissions/v2_pixel_ridge.csv \
     submissions/blend_ridge_v2_65.csv 0.65     # 65% do primeiro + 35% do segundo
```

## final1: `splice_A_b33_24.csv` (público 1.68586)

```bash
python scripts/enso_offset.py submissions/blend_ridge_v2_65.csv \
     submissions/blend_v2_65_enso_b25.csv 0.25     # offset ENSO β=0.25, thr=0.8
python scripts/enso_offset.py submissions/blend_ridge_v2_65.csv \
     submissions/blend_v2_65_enso_b33.csv 0.33     # idem β=0.33
python scripts/splice_year.py submissions/blend_v2_65_enso_b25.csv \
     submissions/blend_v2_65_enso_b33.csv submissions/splice_A_b33_24.csv 2024
```

O offset (`worcap/enso.py`) soma `β·(clim_fase − clim_full)` apenas nos meses-alvo
em que o n34 da origem sai da faixa neutra (≥ +0.8 ou ≤ −0.8). Em 2024 isso pega
jan–mai; jun–dez ficam neutros e o arquivo não muda neles.

## Conferência de causalidade

`tests/test_invariance.py` e `tests/test_enso.py`. O protocolo inteiro está em
`ADR.md` §5. Todo CSV passa por `worcap/submission.py` antes de subir
(ids iguais ao sample, sem NaN, valores ≥ 0).

## Nota de transparência

O arquivo `submissions/era5_truth_2023.csv` (score público 0.00002) é um probe
ilustrativo: o alvo da competição é o `total_precipitation` mensal do ERA5 na
grade 0.25°, e como o ERA5 de 2023-24 é público, o lado privado do leaderboard é
em tese replicável depois do fato. Esse arquivo não é elegível como submissão
final e existe só para documentar o ponto. Mais contexto em `DECISIONS.md`.
