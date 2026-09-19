# Manifesto de dados externos (auditoria)

Todos os dados externos são públicos/gratuitos, baixados programaticamente,
com proveniência e uso documentados. Nenhum dado pós-M−1 é usado como feature.

| arquivo | fonte | endpoint/dataset | uso | licença | md5 |
|---|---|---|---|---|---|
| `data_ext/ersst_v5_sst_mnmean.nc` | NOAA PSL — ERSSTv5 SST mensal 2° | `https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc` | índices oceânicos + PDO (EOF1) | NOAA public domain | `913a5910d4946e095b0fd0221800a467` |
| `data_ext/era5_mslp_southocean.nc` | C3S — ERA5 monthly MSLP, banda −35..−70 | `cdsapi`: `reanalysis-era5-single-levels-monthly-means`, var `mean_sea_level_pressure`, 1940–2024 | SAM-proxy (MSLP zonal 40S−65S) | Copernicus licence | `68971974b87baa50d2953cce12a0cd7b` |
| `data_ext/nmme/ncep_sys2_*_lead1.nc` | C3S — `seasonal-monthly-single-levels`, centro `ncep`, system 2, `tprate`, lead 1 | hindcast 1982–2017 + realtime 2018–2024 | features dinâmicas (anomalia debiased) | Copernicus + non-EU contributions licence | `590a0f22…`, `82e99b5b…` |
| `data_ext/nmme/ecmwf_sys5_*_lead1.nc` | C3S — mesmo dataset, centro `ecmwf`, system 5 | hindcast 1981–2016 + realtime 2017–2024 | idem | idem | `3b218f71…`, `cee52200…` |
| `data_ext/nmme/nmme_raw_grid.nc` | derivado | ensemble-mean regridado p/ grade ERA5 (301×261), mm/dia | artefato intermediário (gerado por `worcap/nmme.py::build_raw_grid`) | — | `10d431a3…` |

## Convenções e decisões de uso

- **ERSSTv5**: anomalias vs climatologia 1940–2022 (período de treino). Caixas e
  derivados em `worcap/indices.py::BOXES`. PDO = PC1 do Pacífico Norte após
  remover média global por timestep (convenção Mantua/Deser); orientação
  PDO+ ⇔ corr>0 com Niño3.4.
- **SAM-proxy**: `zscore(MSLP zonal 40°S) − zscore(MSLP zonal 65°S)`
  (Gong & Wang 1999). Domínio oficial termina em −60°S → ERA5 externo é
  obrigatório para a banda subpolar.
- **NMME lead-1**: init no mês S prevê S+1 (mesma indexação por mês de origem
  da tabela compartilhada). Conversão `m/s → mm/dia` = ×86400×1000.
  Ensemble-mean por modelo; debias = anomalia vs climatologia de inits ≤2016
  **excluindo os anos do fold** (ver `worcap/nmme.py::anom_on_grid`).
- Cobertura NMME: NCEP hindcast 1993–2016 + realtime 2019-10→2024-12;
  ECMWF 1981–2016 + 2017→2024. Inits fora da cobertura → feature 0 (neutro).
- **Causalidade**: todas as features externas usam apenas dados ≤ fim do mês
  M−1 (mesmo critério do teste). Índices de 2023–24 entram via `time_origem`.

## Downloads não utilizados / rejeitados

- Índices NOAA `.long.data` (nino34, atl3, sam, pdo…): downloads vieram como
  páginas de erro HTML — índices recomputados do ERSSTv5 em vez disso.
- OISSTv2: cobertura só 1981+ (treino começa 1940) → ERSSTv5 escolhido.
- IRI Data Library / NCEI ERDDAP / CPC FTP (NMME): indisponíveis sem login ou
  aposentados → C3S usado como fonte única.

*Gerado: 18/09/2026. Hashes completos via `md5sum data_ext/**`.*
