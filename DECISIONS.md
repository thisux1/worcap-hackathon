# DECISIONS — Hackathon WorCAP 2026 (documento vivo)

Log cronológico de **todas** as decisões, descobertas e pendências da competição.
Regra de uso: **toda decisão, achado de pesquisa ou mudança de premissa entra aqui no mesmo dia** — antes ou junto da mudança correspondente no código ou no `ADR.md`. O ADR guarda a arquitetura estável; este arquivo guarda a história e o "porquê" datado.

## Convenção de entrada

- **D-NNN** = decisão tomada · **R-NNN** = achado de pesquisa (repo, paper, ferramenta, post) · **P-NNN** = pendência/confirmação em aberto
- Status: `proposto` | `decidido` | `rejeitado` | `em investigação` | `confirmado` | `resolvido`
- Vereditos de pesquisa (R): `adotar` | `minerar` (copiar técnica/código específico) | `investigar` (spike com deadline) | `referência` | `descartar`
- Formato de entrada: `ID | data | título` → contexto / decisão ou veredito / motivo / fontes
- Competição roda de **15–22/09/2026** — dia 0 = 15/09 (hoje).

---

## Decisões vigentes

| ID | Data | Decisão | Status | Motivo |
|---|---|---|---|---|
| D-001 | 15/09 | Fatos oficiais incorporados à ADR v4: métrica RMSE absoluto (mm/dia), alvo = próprio ERA5, treino 1940–2022, LB público 2023 / privado 2024, CSV `ID={ano}_{mes}_{lat}_{lon}`, ≤3 submits/dia, GPU 30h/sem, auditoria de código | confirmado | Vídeos da organização (Jerônimo, Carlos) |
| D-002 | 15/09 | Corte de V4-CNN e V5-ACE2 (não é adiamento) | decidido | 10 dias, 3 submits/dia, LB armadilhoso, auditoria — custo/benefício negativo |
| D-003 | 15/09 | V3 dinâmico é **condicional**: kill-switch fim do dia 2 → fallback NMME/IRI OPeNDAP → drop | decidido | Filas MARS/CDS de horas–dias inviabilizam maratona curta |
| D-004 | 15/09 | Pipeline enxuto: V0 + V0.5 + V1 + V2 (+V3 condicional) | decidido | Consistência > teto teórico sob prazo apertado |
| D-005 | 15/09 | "CV decide, LB confirma" — submits servem para validar infra e CV, nunca hill-climbing | decidido | Armadilha de 2023: LB público mede regime ENSO distinto do privado 2024 |
| D-006 | 15/09 | Climatologia promovida a componente de modelo (não só baseline): variantes testadas em OOF | decidido | RMSE absoluto faz o erro de `clim` entrar sem atenuação em `ŷ = clim + σ·ẑ` |
| D-007 | 15/09 | Regime de features do teste é assimétrico: 9 vars atmosféricas do mês M−1 fornecidas p/ todos os 24 alvos (`teste_features.nc`, coord `time_origem`); precipitação observada **termina em dez/2022** (`tp_ultima_obs` + `lag_meses` 1–24). Lags de chuva só existem até dez/2022 → 2023 tem lags 1–12, 2024 só lags 13–24. Recursão (previsão própria como entrada) é opção a avaliar em OOF, com risco de propagação de erro | confirmado | Aba Data do Kaggle |
| D-008 | 15/09 | Features externas restritas a **≤ fim do mês M−1** (causal, audit-safe). SST/índices observados do próprio mês-alvo dariam skill de graça (competição é retrospectiva — dados de 2023–24 existem), mas violam o framing "estado de M → prevê M+1" e podem ser lidos como leakage na auditoria | decidido (reversível se P-007 liberar) | Framing oficial da tarefa + auditoria de código |
| D-009 | 15/09 | Baseline oficial implícito = climatologia: a organização declara "a régua a bater é a climatologia" | confirmado | Overview do Kaggle — V0 precisa ser forte, não placeholder |
| D-010 | 15/09 | Cronograma comprimido: Kaggle mostra "start 2 days ago, 7 days to go" → ~7 dias restantes a partir de 15/09 | confirmado | Página da competição |
| D-011 | 15/09 | Submeter candidatos finais **cedo**: §7b — empate no score favorece a submissão mais antiga | decidido | Foundational Rules §7b |
| D-012 | 16/09 | **Lag augmentation** no treino do GBM: no teste a última obs de precip é sempre dez/2022 (lag 1–24); no treino o último obs real seria sempre lag 1 → para cada row de treino sorteia-se lag ~ U{1..24} e a feature `tp_lastobs_a` usa a anomalia do mês correspondente. Avaliação em 2 regimes reais: âncora dez/(Y−1) (lags 1–12, tipo 2023) e dez/(Y−2) (lags 13–24, tipo 2024) | decidido | D-007 — sem isso o modelo aprende persistência lag-1 que não existe no teste |
| D-013 | 16/09 | Anomalias de features usam **climatologia leave-one-out** do próprio ano (mesmo mês-calendário, exclui o ano corrente) — sem leakage e sem custo de refit por fold. Já o alvo z e a reconstrução `clim+σ·ẑ` usam clim/σ fitados **por fold** (só train_years) | decidido | Leakage-safe nas duas pontas |
| D-014 | 16/09 | Macro-regiões para médias regionais de anomalia: Amazonia (−15..5, −75..−50), NEB (−15..−2, −45..−34), Sudeste (−25..−15, −50..−38), Sul/SESA (−35..−22, −60..−48), Atl. tropical (−5..10, −40..−20). Índices × janela sazonal do alvo: nino34×{SON,DJF}, amm×{FMA,MAM}, atl3×{MJJ,JJA}, saod×{DJF}, ep_cp×{SON,DJF} | decidido | Caixas usuais da literatura SA; interações seguem teleconexões conhecidas |
| D-015 | 17/09 | **Computação pesada migra para Kaggle kernels**: laptop com ~2 GB RAM livre e CPU compartilhada não comporta o GBM (matrizes >1 GB, treino ~30 min). Dataset privado `thisux1/worcap-assets` (pacote `worcap/` + parquets) + kernel script privado `thisux1/worcap-v1` com `competition_sources` anexado; outputs baixados via API, submits via `kaggle competitions submit` local. `config.py` portável via env vars | decidido | ~29 GB RAM + 4 vCPU + GPU 30h/sem disponíveis; mesma codebase local/Kaggle |

---

## Descobertas de pesquisa — rodada 15/09

| ID | Item | O que é | Veredito | Uso proposto |
|---|---|---|---|---|
| R-001 | `github.com/paulo-o/forecast_rodeo` | Código do vencedor do Subseasonal Rodeo (MultiLLR + AutoKNN + CFSv2 debiased), Python 2.7 | minerar | Detalhes de implementação do MultiLLR (V0.5) e AutoKNN (V2b) |
| R-002 | `github.com/monego/lgb-prec-unc` | **Autores INPE**: LightGBM p/ precipitação mensal na América do Sul; inputs = variáveis do mês anterior; Optuna; 2ª árvore p/ incerteza; treino 1980–2017, teste 2018–19 | minerar | Referência direta do V1 + ideia de UQ por segunda árvore |
| R-003 | `xeofs` (xarray-contrib) | EOF/PCA em xarray+dask, peso coslat, variantes (rotada, MCA), bootstrap | adotar | EOFs de SST detrended do V2a; possível MCA SST×precip |
| R-004 | `xskillscore` (xarray-contrib) | Métricas de verificação: pearson_r (ACC temporal), CRPS (ensemble/gaussiano), RPS, ROC, rank histogram | adotar | Painel de métricas do §5 da ADR |
| R-005 | `lilio` + `s2spy` (AI4S2S) | Calendário com lags p/ resample + train/test splits leakage-safe p/ S2S | investigar | Spike 2 h no dia 1: se acelerar a montagem de features com lag, adota; senão pandas puro |
| R-006 | `github.com/gscerveira/telnet` | Repo do TelNet com CLI `telnet download --era5 --indices --seas5` | minerar | Pipeline pronto de download de índices climáticos (economiza o dia 1) |
| R-007 | PyCPT (IRI) | Ferramenta canônica de previsão sazonal: NMME+C3S via IRIDL, CCA/MOS, calibração, verificação | referência | Provável pesada demais p/ maratona; consultar p/ desenho de MOS se V3 entrar |
| R-008 | pyNMME (EnvSoft 2023) | Retrieve/calibrate/verify NMME: linear scaling, QM, Bernoulli-Gamma-Gaussian, 20 métricas | minerar | Receitas de debias do fallback NMME |
| R-009 | `github.com/mktippett/nmme-zarr` | Fetch OPeNDAP→zarr local do NMME (com retry/resume) | minerar | V3-fallback: código de download NMME quase pronto |
| R-010 | `climpred` | HindcastEnsemble: verify, ACC, remove_seasonality, alinhamento init×lead | adotar se V3 | Verificação de hindcast do V3 |
| R-011 | `C3S_evaluator` + notebooks C3S (sf-anomalies, sf-verification) | Requests CDS sazonais prontos + cálculo de anomalia + métricas de tercil | minerar | Requests e pós-processamento do V3 |
| R-012 | s2s-ai-challenge (climetlab plugin; `HoratN/pp-s2s`) | Infra do desafio S2S WMO + pós-processamento probabilístico | referência | Padrões de pipeline e benchmarks de calibração |
| R-013 | **FuXi-S2S** (Zenodo ONNX; HF `FudanFuXi/FuXi-S2S`) | Modelo AI S2S 42 dias, 1,5°, treinado em ERA5 — supera ECMWF S2S em precipitação; hindcasts prontos 2002–2021 (não cobre 2023–24 → teria que rodar inferência com ERA5 diário); licença CC BY-NC-ND | **descartar** | Regras §6c: modelo na submissão exige licença OSI sem restrição comercial — NC-ND barra. No máximo diagnóstico comparativo no relatório |
| R-014 | FengWu-W2S | Modelo weather→S2S 42 dias | descartar | Pesos não publicados (repo é reprodução não-oficial) |
| R-015 | Writeups LEAP/ClimSim (1º, 4º, 10º lugares) | Squeezeformer/TPU, dezenas de modelos, pseudo-labels | descartar | Confirma o corte de DL — mas nota boa prática: vencedor publicou pipeline completo reproduzível p/ auditoria |
| R-016 | 1º lugar "Localised Precipitation Forecasting Brazzaville" | LGBM + CatBoost com group-CV + stacker Ridge | referência | Corrobora desenho V1 + stacker ridge |

---

## Pendências (confirmar com organização / no Kaggle)

| ID | Item | Por quê | Status |
|---|---|---|---|
| P-001 | **Tamanho do time: 3 ou 4?** Vídeo do Jerônimo diz ≤4; post oficial do LinkedIn (jul/2026) diz ≤3 | Muda divisão de trabalho | **resolvido 18/09**: Rules oficiais dizem ≤4 (prevalece sobre LinkedIn) |
| P-002 | URL + schema: `kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul` (restrita a inscritos — 404 anônimo). CSV = `id,tp_mm_day`; id = `{ano}_{mes}_{lat}_{lon}` com 2 casas decimais (ex. `2023_01_-60.00_-90.00`); **ordem vem do `sample_submission.csv` — não reconstruir**; 1.885.464 linhas = 24 meses × 78.561; 13 arquivos, 2,06 GB | — | resolvido |
| P-003 | Unidades/grade: alvo e features já em **mm/dia**; lat crescente −60→15, lon −90→−25, 0,25°; domínio inclui oceano e Andes — RMSE sobre **todos** os 78.561 pontos; `treino_tp_alvo.nc` já vem com o shift M+1 aplicado (último mês NaN) | — | resolvido |
| P-004 | Corte/emissão: features oficiais do teste são do mês M−1 (coord `time_origem`) → data de emissão efetiva = fim do mês M−1... atenção: features = mês *anterior ao alvo*, ex. alvo jan/2023 usa dez/2022 | — | resolvido |
| P-005 | Formato exigido de citação de dados externos e escopo da auditoria de código. Rules coladas = só Foundational Rules genéricas; se houver seção de regras **específicas** da competição, checar lá | Manifesto de auditoria precisa casar com o exigido | resolvido |
| P-006 | Licença FuXi-S2S vs. regras | §6c das Foundational Rules exige licença OSI sem limite de uso comercial; CC BY-NC-ND viola | resolvido → R-013 descartado |
| P-007 | Perguntar no Discord: dados externos observados do **próprio mês-alvo** (ex.: SST de jan/2023 p/ prever jan/2023) são permitidos? Default = não usar (D-008) | Se liberado, skill de graça via oceano real | resolvido — regra de causalidade explícita transcrita em RULES.md (D-008 confirmado) |
| P-008 | Campos "Maximum Team Size" e "Submissions per day" no site (sidebar/topo de Rules) + eventual seção de regras específicas. **Submits/dia = 5** (API respondeu "4 submissions remaining" após 1º envio) — vídeo dizia 3, prevalece a API | Fecha P-001 | resolvido — team ≤4 e 5 submits/dia ambos confirmados nas Rules |

---

## Resultados experimentais (OOF / LOYO+embargo ±1)

| Data | Via | Config | OOF RMSE (mm/dia) | Notas |
|---|---|---|---|---|
| 15/09 | V0 | climatologia (mês,pixel), média simples 1940–2022 | **1,8081** | min 1,514 / máx 2,318 por ano; pior Jan–Fev (~2,0, estação úmida), melhor Set (~1,49) |
| 15/09 | V0 | idem + tendência linear por (mês,pixel) | **1,8040** | ganho marginal +0,004; tendência extrapola para 2023–24 — leve risco, revisar antes do final |
| 15/09 | ref | persistência (ŷ = mês anterior) | 2,5751 | climatologia domina, como esperado |
| 16/09 | V0.5 | ridge por pixel, X compartilhado (índices ERSST + médias regionais das 9 vars + calendário + interações; 70 cols), alvo z | λ=300 → 1,7398; λ=1000 → **1,7379**; λ=3000: 1,7485; λ=10000: 1,7706 | **bate a climatologia (−0,070)**; ótimo ~λ=1000, curva plana 300–1000 |
| 16/09 | V1 v0 | LGBM global, 51 cols (sem médias regionais), 6k px/mês, 400 iters | subset 7 anos: lag1-12 **1,850** / lag13-24 **1,857** vs clim 1,959 | bate clim todo ano, mas **perde do ridge** (1,784 no subset) — faltavam as 45 cols regionais |
| 16/09 | V1 v0.5 | idem + 70 shared cols + early stop (ano interno), 5k px/mês | 1983: 1,838 (era 1,877); 1988: 1,622 (era 1,611) | melhora irregular; GBM ainda atrás do ridge — caminho provável = blend ridge+GBM |
| 17/09 | V2 | **mini-Rodeo**: ridge compartilhado (λ1=1000) + ridge por pixel no resíduo (λ2=300) com 22 feats per-pixel (9 anom atm LOO + 9 nbhd + tp_lastobs + lag + lag×tp) | subset 7 anos: lag1-12 **1,7596** / lag13-24 **1,7618** vs ridge 1,7816 | **−0,022 em TODOS os anos e regimes** → features per-pixel ajudam via coeficiente linear local (o que o GBM não conseguiu); kernel Kaggle, ~4 min/fit |
| 17/09 | V0.5 rerun | ridge λ=1000 com tabela 108 cols (OOF store regenerado limpo) | **1,7391** | ≈ idem às 70 cols (1,7379) — expansão de features neutra no OOF |
| 16/09 | — | ERSSTv5 processado → `data_ext/indices_ersst.parquet`: nino12/3/34/4, tna, tsa, atl3, tio, saod_n/s, pmm_n, **npac** (proxy PDO: 20–60°N×150–210°E), amm, saod, ep_cp, pdo_x_nino34. Sanity: nino34 dez/2015=+2,85, nov/2023=+2,31; tna jul/2023=+1,58 | — | lat ERSST descendente (bug corrigido); downloads NOAA `*.long.*` vieram como páginas de erro → npac substitui PDO oficial |
| 16/09 | — | `features_shared.parquet` expandida p/ 108 cols: +npac, pdo_x_nino34, lags {1,2,3} e médias móveis 3m de 9 índices | — | ridge OOF com 108 cols: rerun interrompido no meio — OOF store de v05 ficou **misto** (folds <1979 usam 108 cols) — regenerar antes de conclusões finais |

Submissões geradas e validadas (schema, 1.885.464 linhas, ids idênticos ao sample, sem NaN, ≥0):
`submissions/v0_clim.csv`, `submissions/v0_clim_trend.csv`. Ordem dos ids confirmada = C-order (time→lat→lon).

### Log de submissões

| Data (UTC) | Arquivo | LB público | OOF | Notas |
|---|---|---|---|---|
| 16/09 21:03 | v0_clim.csv | **1.85077** | 1.8081 | 1º submit; gap CV→LB só +0,04 → protocolo LOYO representativo p/ 2023 |
| 17/09 02:41 | v05_ridge.csv | **1.72679** | 1.7379 | λ=1000, fit 1940–2017. LB veio −0,011 abaixo do OOF → CV honesto; **−0,124 vs climatologia** |
| 17/09 08:59 | v1_lgbm.csv | 1.84438 | — | LGBM 20k px/mês, 134 cols, lag augmentation, best_iter=674 (kernel Kaggle v9, 27 min eager). Só −0,006 vs climatologia → GBM **não** é a via; erro altamente correlacionado ao ridge não justifica blend forte |
| 17/09 09:01 | v05_ridge_all.csv | **1.69966** | — | λ=1000 refit em **1940–2022** (+5 anos vs submit anterior). −0,027 vs fit 1940–2017 → anos recentes ajudam (regime mais próximo do teste). **Melhor LB até agora: −0,151 vs climatologia** |
| 17/09 10:34 | v2_pixel_ridge.csv | 1.70886 | subset 1,760 | mini-Rodeo λ2=300, fit 1940–2022. CV melhorou em todos os anos (−0,022) mas LB **piorou +0,009** vs ridge_all → ganho per-pixel não transferiu p/ 2023; sinal local é real porém pequeno. Candidato a λ2 maior / feature set menor |

**D-012 — modelo final = família linear, não GBM.** LB confirma CV: GBM (1.844) fica muito atrás do ridge (1.727/1.700) — o sinal está nas features mensais compartilhadas exploradas linearmente por pixel, não em interações não-lineares por pixel. GBM fica como membro minoritário de ensemble, no máximo.

**D-013 — refit em todos os anos para submits finais.** Holdout 2018–2022 serviu para seleção; variante escolhida → refit 1940–2022 ganha −0,027 de LB. Prática padrão e honesta (seleção continua via OOF).

**Pendente:** OOF store de v05 regenerado ✓ (1,7391 c/ 108 cols). Próximos candidatos: (a) V2 com λ2 maior (1000/3000) — sinal per-pixel é real mas frágil no LB; (b) média ridge_all+v2 (decorrelação parcial pode ajudar); (c) **via dinâmica SEAS5/NMME** — única que enxerga o estado real de 2024, maior upside restante; (d) blend com GBM descartado (LB 1,844, erros correlacionados).

---

## Índice de fontes

- Organização WorCAP 2026: vídeos de abertura (Jerônimo; Carlos), palestra Dra. Marília, post LinkedIn oficial (jul/2026), Discord do evento
- Rodeo: Hwang et al. KDD'19 (arXiv 1809.07394) + repo `paulo-o/forecast_rodeo` + página L. Mackey (lmackey.github.io/forecastrodeo.html)
- S2S AI Challenge: s2s-ai-challenge.github.io + `ecmwf-lab/climetlab-s2s-ai-challenge` + `HoratN/pp-s2s`
- Pacotes: xeofs, xskillscore, climpred, lilio/s2spy (AI4S2S), C3S_evaluator, PyCPT (iri-pycpt.github.io), nmme-zarr
- NMME: IRIDL `iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/` (OPeNDAP, xarray `decode_times=False`)
- INPE/ML-SA: `monego/lgb-prec-unc` (Atmosphere 2022), `gscerveira/telnet`, arXiv 2512.13910
- AI S2S: FuXi-S2S (Nat. Comms. 2024; Zenodo ONNX; HF `FudanFuXi/FuXi-S2S`, 2002–2021), FengWu-W2S (arXiv 2411.10191 — sem pesos)
- Kaggle comps correlatas: LEAP/ClimSim writeups (jul/2024), Localised Precipitation Brazzaville

---

*Última atualização: 16/09/2026 — próxima entrada esperada: resultado do grid λ estendido do V0.5 e OOF do V1-LGBM nos regimes lag1-12 / lag13-24.*

---

## Sessão 4 — via dinâmica NMME/SEAS5 (18/09)

### Aquisição e cobertura

Fonte adotada: **C3S `seasonal-monthly-single-levels`** via `cdsapi` (IRI agora exige login; NCEI aposentou os datasets NMME; CPC FTP só tem páginas de verificação). Var `tprate`, leadtime_month=1, região SA.

Cobertura real descoberta:

| modelo | hindcast | realtime | membros |
|---|---|---|---|
| NCEP CFSv2 (system 2) | 1993–2016 | 2019-10→2024-12 | 28 / 124 |
| ECMWF SEAS5 (system 5) | 1981–2016 | 2017→2024 | 25 |

→ ECMWF cobre o gap pré-1993 e 2017–2019; **todos os inits do teste (dez/2022–nov/2024) presentes nos dois**.

### Bug de unidades (corrigido)

`tprate` vem em **m/s**. Conversão correta: `×86400 s/dia ×1000 mm/m`. A primeira avaliação usou só ×86400 (m/dia → valores 1000× menores) — todos os números daquela rodada descartados.

### Skill honesto pós-correção (lead-1, hindcast 1994–2016, debiased por mês-init)

| config | RMSE | vs clim ERA5 (1.4652) |
|---|---|---|
| NMME-deb MME-mean | 1.6001 | **+0.135 pior** |
| NCEP-deb | 1.6254 | pior em 90.7% dos px |
| ECMWF-deb | 1.6197 | pior em 98.6% dos px |

→ **Dinâmico standalone perde feio da climatologia em RMSE** (previsão suavizada adiciona variância de erro; corr anomalia média ~0.12). O resultado anterior "74.5% dos px melhores" era artefato da escala 1000× menor.

**D-014 — NMME/SEAS5 entra como FEATURE, não como previsão.** O sinal existe (corr ~0.12 média, físico: El Niño 2015 → anomalia seca Amazônia/NEB, úmida Sul) mas a amplitude precisa ser calibrada pelo modelo — exatamente o que ridge faz. Integração em 2 níveis: (a) 5 cols `nmme_<reg>_an` na shared table (113 cols); (b) campo per-pixel `nmme_an` no stage-2 do V2 (K2=23). Clim do modelo sempre exclui anos fora do fold (LOYO-safe).

### Estado da integração

- `worcap/nmme.py`: `_norm` (dims→init, m/s→mm/dia), `build_raw_grid` (regrid por chunks p/ grade ERA5 — evita OOM), `anom_on_grid` (debias hindcast ≤2016 excluindo anos do fold), `region_table`, `load_raw(model)` p/ comparação por centro.
- `nmme_raw_grid.nc` (317MB): ecmwf+ncep, (528,301,261) cada.
- `pixel_ridge.py`: `use_nmme=True` → `K2_NMME=23`, `_nmme_map` LOYO-safe.
- `features_shared.parquet`: +5 cols nmme → **113 cols**.
- Kernel v17 rodando: eval V2 vs V2+NMME em {1997,1998,2005,2010,2015} × 2 regimes de lag.

*Última atualização: 18/09/2026 — próxima entrada: resultado do eval NMME (kernel v17) e decisão de submit.*

### Resultados da sessão 4 (continuação)

**SAM via ERA5-CDS (executado):** `era5_mslp_southocean.nc` (234MB, MSLP mensal −35..−70, 1940–2024). SAM-proxy = zscore(MSLP zonal 40S) − zscore(MSLP zonal 65S) — cobertura completa 1940+ (melhor que Marshall/BAS que começa 1957). **SAM em dez/2023 = +4.28σ** — regime fortemente positivo exatamente no período de teste.

**PDO via EOF1 (executado):** PC1 do Pacífico Norte (ERSST, convenção Mantua/Deser) — 11.6% var, corr(nino34)=+0.465, PDO+ no El Niño 97–98 ✓. Substituiu `npac` no termo `pdo_x_nino34`.

**Eval V2+NMME (kernel v18, era NMME 1997–2015, média 5 anos):**
- nmme=False: gap1 1.8084 / gap2 1.8090
- nmme=True: gap1 **1.8037** / gap2 **1.8046** → **−0.005**
- Padrão: ganha nos anos de transição ENSO (1998: −0.018; 2010: −0.008), perde migalhas em neutros. **D-015: sinal dinâmico real mas pequeno — manter como feature, nunca standalone.**

**OOF ridge λ=1000 com 124 cols (SAM+PDO+NMME-regional): 1.7391** — idêntico ao baseline 108-col. Quebra por era: 1940–92 = 1.7424, 1994–2016 = 1.7362 (NMME não piora; efeito diluído). SAM/PDO inócuos no OOF — mantidos por motivação física e possível valor no regime +4σ do teste.

**Submissão gerada:** `v05_sam_pdo_nmme.csv` (ridge_all refit 1940–2022, 124 cols) — limite diário atingido (slot abre ~21:03 UTC).

### Submits de 18/09 (5/5 usados — reset = meia-noite UTC, não janela móvel)

| arquivo | LB público | leitura |
|---|---|---|
| v05_sam_pdo_nmme.csv | 1.73220 | ridge_all + 124 cols (SAM/PDO/NMME-reg). **+0.033 pior** que ridge_all → cols novas prejudicam 2023 |
| blend_ridge_v2.csv (60/40) | **1.69836** | **novo melhor LB** (−0.0013 vs ridge_all). Blend confirma decorrelação marginal |
| v2_nmme.csv | 1.75175 | +0.043 vs v2 puro → NMME per-pixel NÃO transferiu p/ 2023 (CV dizia −0.005!) |
| blend_ridge_v2_50.csv (50/50) | 1.69908 | curva de peso: ótimo ~0.6–0.7 no ridge |
| blend_sp20.csv (80/20) | 1.70403 | entre os pais, como esperado |

**D-016 — features novas (SAM/PDO/NMME) reprovadas no LB.** Apesar de fisicamente motivadas e OOF-neutras, todas as variantes com a tabela 124-col degradaram vs contrapartes 108-col. Mecanismo provável: cols NMME=0 em ~metade do treino distorcem a padronização do ridge + overfit de regime que o LOYO não captura (2023 ≠ média histórica). **Decisão: feature set congelado em 108 cols; pipeline NMME arquivado** (custou ~1 dia de computação Kaggle — sunk, não insistir).

**D-017 — blend como via principal.** Curva LB: 100/0=1.69966, 60/40=1.69836, 50/50=1.69908 → ótimo ~0.60–0.65 ridge. Ganho pequeno mas consistente com CV (v2 melhor em CV, pior em LB → combinação amortece).

**Bug corrigido:** `v2_nmme.csv` saiu com 53.568 NaN (4.464 px × 12m) — `to_array` alinhando modelos com coberturas distintas + borda de interpolação ECMWF → NaN propagou. Fix: `np.nan_to_num` em `_nmme_map`/`region_table`/`anom_field`.

**Overnight:** kernel v21 avalia λ2 ∈ {1000, 3000} no V2 (tabela 108-col) — se regularização maior estabilizar o sinal per-pixel, vira membro de blend melhor.

*Última atualização: 18/09 00:10 UTC — melhor LB: 1.69836 (blend 60/40). Restam 5 submits/dia a partir de agora.*

### Sessão 5 — λ2 e damping (18/09 ~02:00 UTC)

**λ2 do V2 (kernel v21, tabela 108-col, 7 anos × 2 regimes):** λ2=300 já era ótimo — monotônico: 300→(1.7596/1.7618), 1000→(1.7612/1.7637), 3000→(1.7631/1.7656). V2 congelado como membro de blend.

**Damping (descoberta):** encolher a componente anômala da previsão — `pred' = clip(clim + γ(pred−clim), 0)` — avaliado no OOF store do v05: γ ótimo = **0.80** (RMSE 1.7406 vs 1.7442 em γ=1; mínimo em 0.80, curva suave). Interpretação: o ridge sobrestima a amplitude das anomalias (~20%) — overfit leve, corrigível sem custo. `scripts/damp.py` aplica a qualquer CSV.

**Fila de submissões para 19/09 (reset 00:00 UTC):**
1. `blend_v2_d80.csv` — blend 60/40 + damping 0.80 (top candidato; esperado ~1.694)
2. `blend_v2_65_d80.csv` — peso 65/35 + damping
3. `blend_ridge_v2_65.csv` — 65/35 sem damping (isola efeito)
4. `v05_ridge_all_d80.csv` — ridge damped sozinho (candidato a final submission)
5. reserva conforme resultados

**Bug do OOF store descoberto no caminho:** `time` no parquet OOF = mês-ALVO (não origem) — a primeira análise de damping usou clim deslocada +1m (RMSE 2.21 espúrio). Corrigido: `orig = T − 1mo`, `clim[T.month]`.

*Última atualização: 18/09 02:05 UTC — melhor LB: 1.69836. Próximos submits só após 00:00 UTC 19/09.*

### Sessão 6 — Rules oficiais transcritas (18/09 ~21:30 UTC)

Aba Rules copiada para `RULES.md` + esclarecimentos do organizador no canal oficial. Fecha P-001
(equipe ≤4), P-007 (dado externo do mês-alvo **proibido** — "qualquer dado disponível até fim de
T−1, nada posterior"; nosso pipeline já indexa por `time_origem`=T−1, compliant por construção).
Novidades operacionais: dados fora do domínio e em escala diária/semanal explicitamente legais;
desempate premia submissão mais antiga; código do vencedor deve ser OSI sem limite comercial →
adicionado LICENSE MIT. Sem feature obrigatória: formato de saída + causalidade + auditabilidade
são as exigências reais.

**Canal #duvidas transcrito em RULES.md.** Confirmações oficiais: Copernicus/SEAS5 com cadastro
gratuito = dado público OK; autorregressão legal; dados semanais/diários e fora do domínio legais;
comissão avalia além da métrica; sem pitch; entrega = acesso ao código. **Alerta:** o leak do
`teste_features` (campos atmosféricos do alvo aparecem na linha seguinte) foi reportado
publicamente — organização respondeu que já sabia e **vai revisar as submissões** por leakage.
Nosso pipeline usa só `time_origem`=T−1 — `tests/test_invariance.py` criado: prova empírica de que
a previsão de T depende apenas de dados ≤ T−1 (4/4 checks verdes).

### Sessão 7 — Auditoria por subagentes + vereditos (18/09 ~23:30 UTC)

**Auditoria de leakage (subagente):** pipeline 108-col limpo. Achados: (a) **PDO fitava transform em dados de teste** (SVD/stats/sign sobre série completa) — corrigido: fit ≤2022, projeta 2023-24 em loadings congelados (`indices.py:93-107`); (b) parquet em disco é 124-col e scripts locais não filtravam → criado `features.production_cols()` (108 exatas, mantém `pdo_x_nino34` original) aplicado nos submits locais; (c) publication lag ERSST/ERA5 (~dias) defensável sob interpretação por timestamp — endossada pela organização, documentar no manifesto; (d) bug menor não-leakage: máscara das interações usa mês de origem em vez do alvo (`features.py:111`).

**Veredito recursão:** implementar spike OOF — única alavanca específica pro regime 2024 (canal tp_lo morto em lag 13-24). Teto medido: Δgap1−gap2=0.0022 → upside ~0.002-0.005. Kill-criterion: só adotar se gap-2 melhorar ≥0.003 E em ≥6/7 anos. Variante R2 (2024-only) preserva público idêntico. `eval_year_rec` implementado — kernel v22 mede.

**Veredito finais (subagente estratégia):** erro ridge×v2 correla ρ≈0.99 → best-of-2 deve segurar as pontas. Final 1 = `blend_v2_d80` (60/40+γ0.80), Final 2 = `v05_ridge_all_d80` (ridge puro+γ0.80, cobre o modo de falha documentado do v2). **Jogada nova:** splice por ano-alvo (`scripts/splice_year.py`) — linhas 2023 da melhor config pública + 2024 da config privada-tilted, custo zero no LB e mantém ambas finais no top-2 público (auto-seleção). Grid (w×γ)×{gap1,gap2} no kernel v22 decide se as linhas 2024 merecem tilt.

**Fila 00:00 UTC (5 submits):** blend_v2_d80, blend_v2_65_d80, blend_ridge_v2_65 (sem damping), v05_ridge_all_d80, blend_v2_d70.

*Última atualização: 18/09 23:40 UTC — LB público: 5º (1.69836). Top-2 (1.499/1.576) estatisticamente suspeitos pós-revelação do leak — revisão anunciada pode subir nosso rank real.*

### Sessão 8 — Resultados 19/09 00:00 UTC + damping rejeitado

| submit | LB | leitura |
|---|---|---|
| **blend_ridge_v2_65** (65/35, sem damping) | **1.69816** | novo melhor |
| blend_v2_d80 | 1.70584 | damping +0.0075 vs undamped |
| blend_v2_65_d80 | 1.70599 | damping +0.0078 |
| v05_ridge_all_d80 | 1.70885 | damping +0.0092 |
| blend_v2_d70 | 1.71410 | γ menor = pior ainda |

**Damping γ=0.80 REJEITADO no LB** — OOF dizia −0.004, LB diz +0.008~0.016: divergência de regime
idêntica à das features NMME. Hipótese: no regime 2023 o modelo já tende a subestimar anomalias
(transição ENSO extrema), então encolher piora. Splice 2024 ainda pode usar γ se grid gap-2 do
kernel v22 mostrar γ*<1 específico do regime lag-longo. **scripts/report.py** criado — dashboard
6 painéis (submits, LB, OOF-vs-LB, damping, blend, status) → report.png.

*Última atualização: 19/09 01:05 UTC — melhor LB: 1.69816 (5º). Kernel v22 rodando.*

### Sessão 9 — Kernel v26 + sweeps: 3 hipóteses mortas, config confirmado

**Recursão (R2): MORTA** pelo kill-criterion pré-registrado. gap2 vs rec por ano:
1983 −0.004, 1988 +0.000, 1997 +0.003, 1998 +0.008, 2005 −0.006, 2010 −0.007, 2015 +0.010
→ média −0.0005 (ruído), só 3/7 melhoram, degradações até +0.010. Critério (≥0.003 e ≥6/7) falhou.

**Janela de climatologia: MORTA.** full é melhor — 30a: +0.026 pior, 20a: +0.046 pior, trend ≡ full.
Registro completo 1940-2022 vence (mais amostras ENSO > adaptação de tendência fraca).

**w ótimo diverge OOF vs LB:** OOF prefere w→0.4 (mais v2) em todo o grid; LB prefere 0.65 (mais ridge).
V2 ajuda "em média" mas prejudica em regime extremo (2023). LB é a verdade pro público — mantemos 0.65.

**γ=1.0 ótimo em TODO o grid OOF** (γ<1 piora em todo w, ambos gaps) — damping morto nas duas métricas.

**Lag ≠ fator:** RMSE lag 13-24 ≈ lag 1-12 (erro dominado por mês-calendário, não staleness) →
splice por lag sem fundamento; tp_lo contribui pouco em qualquer lag.

*19/09 ~05:30 UTC — exp-lambda rodando (λ1×λ2). Próximo: batch 2 (winsorize ±4σ, X2 padronizado, γ>1 em anos-análogos, clim ENSO-condicional).*

### Sessão 10 — Bateria de experimentos (kernels exp-*) + regra pré-registrada das finais

*20/09 ~01:30 UTC. Resultados de 4 kernels paralelos sobre o branch `exp/integ`.*

**Resultados (OOF LOYO, média 7 anos, gap1/gap2):**

| alavanca | resultado | veredito |
|---|---|---|
| **Offset ENSO-condicional** (β·[clim_fase−clim_full], fase por nino34 do mês-origem) | β=0.25/thr=0.8: 1.7499 gap1 / 1.7519 gap2 vs baseline 1.7593/1.7615 (−0.009); por ano: 1983 −0.037, 97 −0.015, 98 −0.018, 2015 −0.011, neutros ~0 | **ADOTADO** — único lever que melhora concentrado no regime-alvo sem custo nos neutros |
| **γ=1.10 inflação** (w=0.65) | −0.003 gap1 e gap2 (grid 0.9–1.2, ótimo interior) | candidato — decide-se pelo LB + gap2 |
| λ1 sweep | 1000 confirmado ótimo | mantém |
| λ2=30 (vs 300) | −0.002, ótimo na BORDA do grid, só 4 anos | suspenso — grid estendido antes de qualquer slot |
| winsorize ±4σ | +0.003 | morto |
| x2std | +0.003 | morto |
| pooling W2 (α<1) | α=1.0 ótimo | morto |
| recursão, damping, clim-window, lag-splice, NMME | — | mortos em sessões anteriores |

**Bug corrigido em submit():** `phase_of()` não recebia `enso_thr` — origens jan/fev-2023 seriam "frias" (La Niña) no path wired vs "neutras" no script. Fix: `phase_of(v, enso_thr, -enso_thr)`.

**Ordem pós-hoc fixada:** `damp(γ) → enso_offset(β)` — offset mantém magnitude β calibrada (γ não infla o offset).

**Extremos do offset verificados:** |Δ|max ~35 mm/dia só na costa Pacífico Equador/Chocó (clim 30-67) — assinatura física canônica El Niño, coerente espacialmente. N=8-22 anos/célula (mês,fase).

**REGRA PRÉ-REGISTRADA de seleção das finais (registrada ANTES de ver os scores do lote A/C/D):**
- Δ_LB(A=enso_b25) ≤ −0.004 → ENSO confirmado → final1 = C se C≥A senão A
- |Δ(A)| < 0.003 → inconclusivo; downside gated (offset≡0 em neutro) → ship via OOF-análogo
- Δ(A) ≥ +0.008 → mata alavanca (falhou no regime favorável) → final1 = blend65 puro
- final2 = `v05_ridge_all` (1.69966, já selecionável, estruturalmente diverso) ou splice `[2023:* | 2024: ridge+offset]` se A confirmar
- Proibido re-tunar β/thr/γ no LB (hill-climb no regime errado)
- γ decide-se por gap2-OOF + C−A, nunca pelo público isolado
- Finais divergem SÓ na metade 2024 (score final = privado; linhas 2023 irrelevantes ao prêmio)

### Sessão 11 — LB confirma offset ENSO (−0.0123); γ global morto; splices privados

*20/09 ~23:00 UTC — 5/5 submits usados.*

**Resultados LB público (vs baseline blend65/35 = 1.69816):**

| submit | score | Δ | leitura |
|---|---|---|---|
| blend+enso β=0.25/thr=0.8 (A) | **1.68586** | **−0.0123** | ENSO CONFIRMADO (gate −0.004 passou 3×) |
| A + γ=1.10 global (C) | 1.68847 | −0.0097 | γ global custa +0.0026 → morto no público |
| blend+enso β=0.15 (D) | 1.68963 | −0.0085 | dose-resposta monotônica |
| splice[A-2023‖ridge_all+enso-2024] (S1) | 1.68586 | =A | mecânica splice validada (público idêntico) |
| splice[A-2023‖A+γ1.10-quente-2024] (S2) | 1.68586 | =A | idem |

**Kernel gphase — γ por fase ENSO (w=0.65, OOF):**
QUENTE: γ>1 melhora monotônico até 1.2 (1.8663 vs 1.8753); NEUTRO/FRIO: γ=1.0 ótimo.
Por ano: 1983 −0.018, 1997 −0.016, 1998 −0.017; 1988/2005/2010 ~0; 2015 prefere γ<1.
→ γ não morreu, é **condicional à fase**; aposta privada S2 = γ só em meses quentes de 2024 (decay ≈ 1983/1998).

**Auditoria por 3 subagentes (sessão de validação):**
- CSVs submetidos verificados por amostragem direta — splice correto (jan-mai/24 inflados, jun-dez = A)
- Bug residual: kaggle_assets/pixel_ridge.py sem enso_thr → sincronizado, dataset v12
- Ordem γ×offset em S1 deviava da registrada (γ inflava offset ×1.1) → regenerado splice_A_gwarm24_v2
- Física: offset sub-estima amplitude real de 2023-24 (costero+Atlântico ausentes do n34) → headroom positivo
- N por célula (mês,fase): 8-22; extremos ±9 = Chocó/costa Equador (físico)

**Lacuna identificada:** 2016 (decay do evento 2015-16, o mais análogo a 2023-24) nunca entrou no eval → gphase v2 rodando com 2016+1992. Decisivo para S2.

**β por ano (kernel enso, thr=0.8):** anos El Niño/decay querem β~0.4-0.5 (1983: −0.054 em β0.5); agregado 0.25 é puxado por anos frios. Em 2024 só meses quentes disparam → β=0.3-0.35 no lado-2024 defensável via OOF-análogo.

**Fila amanhã (5 slots):** γ-quente full-file em A (mede jul-dez/23), ridge_all+offset full-file (valida S1-2024), S2 ordem-corrigida, splice β0.35-2024, reserva.

*Nova melhor LB: 1.68586 (~6º nominal). final1 provável = A; final2 entre S1/S2.*
