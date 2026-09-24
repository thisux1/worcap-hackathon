# Reprodução das submissões finais — equipe (Kaggle): **thisux1**

Pipeline causal: todas as features usam apenas dados ≤ fim do mês de origem (T−1).
Validação: LOYO com embargo ±1 ano; compósitos ENSO e climatologia refitados por fold.

## Dados de entrada
- Oficiais (.nc na raiz): `treino_tp.nc`, `treino_tp_alvo.nc`, `teste_features.nc`, demais `treino_*.nc`
- Externos (`data_ext/`): ver `DATA_MANIFESTO.md` — ERSSTv5 (NOAA), ERA5-MSLP (C3S), NMME (C3S), todos com link e licença

## Setup
```bash
conda env create -f environment.yml && conda activate worcap
export WORCAP_DATA=. WORCAP_EXT=data_ext WORCAP_SUB=submissions
python -m worcap.indices                                  # índices oceânicos causais (mês origem)
python -c "from worcap.features import build_shared_table; build_shared_table()"
```

## final2: `blend_ridge_v2_65.csv` (público 1.69816)
```bash
python scripts/v05_ridge.py --submit --all      # stage-1: ridge global por pixel, lam=1000, fit 1940-2022
python scripts/v2_pixel_ridge.py                # stage-2: correção ridge por pixel, lam2=300
python scripts/blend.py submissions/v05_ridge_all.csv submissions/v2_pixel_ridge.csv \
     submissions/blend_ridge_v2_65.csv 0.65     # 65% stage-1 + 35% stage-2
```

## final1: `splice_A_b33_24.csv` (público 1.68586)
```bash
python scripts/enso_offset.py submissions/blend_ridge_v2_65.csv \
     submissions/blend_v2_65_enso_b25.csv 0.25            # offset ENSO causal β=0.25, thr=0.8
python scripts/enso_offset.py submissions/blend_ridge_v2_65.csv \
     submissions/blend_v2_65_enso_b33.csv 0.33           # idem β=0.33
python scripts/splice_year.py submissions/blend_v2_65_enso_b25.csv \
     submissions/blend_v2_65_enso_b33.csv submissions/splice_A_b33_24.csv 2024
```

Offset ENSO (`worcap/enso.py`): `ŷ += β·(clim_fase − clim_full)` só em meses-alvo
cuja fase no mês de ORIGEM (n34 ≥ +0.8 quente / ≤ −0.8 frio) é não-neutra —
jan–mai/2024 recebem correção El Niño, jun–dez/2024 são neutros (offset ≡ 0).

## Verificação de causalidade
`tests/test_invariance.py`, `tests/test_enso.py`; protocolo completo em `ADR.md` §5.
Cada CSV passa por `worcap/submission.py` (schema, ids, sem NaN, ≥0).

## Nota de transparência
`submissions/era5_truth_2023.csv` (score público 0.00002) é um **probe ilustrativo**:
o alvo é ERA5 `total_precipitation` mensal na grade 0.25°, e ERA5 2023-24 é público —
o privado é portanto teoricamente replicável ex-post. O arquivo não é elegível à
seleção final e existe apenas como evidência desse furo. Detalhes: `DECISIONS.md` §12.
