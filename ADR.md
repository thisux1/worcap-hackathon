# Registro de Decisão Arquitetural (ADR v4)

**Status:** Proposto — substitui a ADR v3 (incorpora as respostas oficiais da organização — vídeos do Jerônimo e do Carlos — e o enquadramento físico da palestra da Dra. Marília)

**Contexto:** Hackathon WorCAP 2026 (INPE), 15–22/09 — previsão da chuva média do mês seguinte (t+1) sobre a América do Sul. **Alvo = o próprio ERA5** (reanálise, não observações). **Métrica = RMSE absoluto sobre precipitação média em mm/dia**, sobre **todos** os pontos de grade e todos os meses do teste. Treino fornecido: **1940–2022**; leaderboard público = **2023**, privado/final = **2024** — 24 meses num único CSV de **1.885.464 linhas**, colunas `id,tp_mm_day`, `id = {ano}_{mes}_{lat}_{lon}` (2 casas decimais, ordem do `sample_submission.csv` — não reconstruir). **Máx. 5 submissões/dia** (confirmado via API), times de até 4 pessoas (confirmar — ver P-001 em DECISIONS.md), 30 h/semana de GPU Kaggle. Dados externos permitidos com citação e **auditoria de código dos finalistas**.

**Dados oficiais (aba Data):** ERA5 mensal 0,25°, 301×261 pontos, domínio 60°S–15°N × 90°–25°O (**inclui oceano** — RMSE cobre pixels oceânicos). Além da precipitação, **9 variáveis atmosféricas** no mesmo grid: t2m, cloud cover, surface pressure, q/rh/T/Z e U/V em 850 hPa. `treino_tp_alvo.nc` já vem com o shift M+1 aplicado (último mês NaN). `teste_features.nc` traz, para cada alvo, os campos do **mês M−1** (coord `time_origem`) + `tp_ultima_obs` (dez/2022) + `lag_meses` (1–24): **precipitação observada termina em dez/2022 — não existe lag de chuva recente no teste**. "A régua a bater é a climatologia" (Overview oficial).

---

## 1. Premissas que dominam o desenho

- **N dobrou — mas continua pequeno.** A série fornecida vai de **1940 a 2022: 83 anos × 12 ≈ 996 amostras mensais por pixel** (a v3 estimava ~540). Por mês-calendário, **N≈83** (~15–17 realizações de ENSO, não ~12). Isso quase dobra a robustez de GBM, EOFs, baselines sazonais e do debias paramétrico do V3; ainda inviabiliza transformer/DL cru. Pooling multi-pixel continua ganhando ~10–30× de informação (autocorrelação temporal + dependência espacial), não ~300×.
- **A grade é grande e a submissão é massiva.** 0,25°×0,25°, **301×261 = 78.561 pixels/mês** (inclui oceano) → CSV único com **1.885.464 linhas** (24 meses), `id,tp_mm_day`, lat/lon com 2 casas decimais, **ordem do sample_submission** — não reconstruir IDs. I/O particionado e validação de schema viram itens de primeira ordem (§6).
- **O teste não tem precipitação — o regime de features é assimétrico.** Para cada alvo só há as 9 variáveis atmosféricas do mês M−1; a precipitação observada para em dez/2022. Alvos de 2023 têm lags de chuva 1–12; os de 2024, só 13–24 — a persistência ancora em `tp_ultima_obs` e decai com `lag_meses`. Recursão (previsão própria como input para 2024) é opção a avaliar em OOF com cuidado de propagação de erro. **Dado externo real de 2023–24 existe** (competição em set/2026): features externas ficam restritas a ≤ fim de M−1 por segurança de auditoria (DECISIONS D-008; usar dado do próprio mês-alvo só se a organização liberar — P-007).
- **RMSE absoluto muda o centro de gravidade do erro.** A métrica é sobre mm/dia, não sobre anomalia. Como a saída é `ŷ = clim + σ·ẑ`, **a climatologia deixa de ser baseline e vira o maior componente do modelo** — erro na média histórica de um pixel tropical chuvoso na estação úmida degrada o RMSE diretamente. O refit de climatologia por fold é o ponto mais sensível a leakage do pipeline inteiro (§4-V0, §5).
- **O alvo é o próprio ERA5.** Não há correção de viés contra observações a aprender: o problema é capturar a dinâmica interna de persistência/transição do ERA5. CHIRPS/MSWEP/estações perdem o papel corretivo — no máximo parágrafo de diagnóstico no relatório.
- **Previsibilidade vem do oceano — confirmado pela física.** Dra. Marília: a atmosfera perde memória em ~10–14 dias (regime caótico); o oceano, pela inércia térmica, retém memória por meses → **SST global e teleconexões oceânicas são a âncora correta**. Nuance regional dela: Amazônia/Centro-Oeste dominados por convecção termodinâmica local rápida (subgrade, difícil de parametrizar → teto de skill baixo); Sul responde com clareza a forçantes dinâmicas frontais e ENSO → **segmentar métricas por macro-região e regime não é cosmético, é física**.
- **A armadilha de 2023 é desenhada.** LB público = 2023 (transição rápida La Niña → Super El Niño); privado = 2024 (dinâmica distinta). Afinar hiperparâmetros ou pesos de stacker no LB público = queda livre no privado. **CV temporal (LOYO + embargo) decide; LB só confirma** (§5).
- **O teste é 100% real-time para todos os dinâmicos.** 2023–2024 está além de qualquer hindcast C3S/NMME → a calibração treinada em hindcast precisa transferir para realtime (membros e climatologias de referência diferentes). Risco que era hipótese na v3 agora é certeza — reforça o debias paramétrico conservador e o kill-switch (§4-V3).
- **Previsões dinâmicas sazonais prontas continuam sendo a feature externa mais forte** (lição do *Subseasonal Climate Forecast Rodeo*: MultiLLR + AutoKNN + dinâmico debiased; precipitação ganha por MLP ensemble sobre features oceânicas incl. SSS) — mas com custo operacional alto sob prazo de maratona: tratada como via **condicional**, não como espinha dorsal (§4-V3).

### Referências diretas ao problema exato (fact-checked)

- **TelNet** (Pinheiro & Ouarda, *Comms. Earth & Env.* 2025): seq2seq interpretável para N pequeno. **Ressalvas:** avaliado só no Ceará e é sazonal (estações 3-meses); ~16 GB VRAM, dias de sampling. Justificativa de design, não via.
- **Augmentation com CMIP6** (mesmos autores, *Atmos. Research* 2025): supera ERA5 e dinâmicos SOTA sobre a SA — era o que tornaria CNN viável. **Cortado** (§4, escopo).
- **arXiv 2512.13910** (autores INPE, dez/2025): LSTM > XGBoost > CNN1D/GRU > BAM, com XAI. **Ressalva:** avaliação só em 2019 — evidência fraca para ranking arquitetural.
- **ACE2**: 450M params, SST persistida nativa, ~1500 anos simulados/dia. **Cortado** (§4, escopo).

---

## 2. Índices e teleconexões — tabela expandida e **sazonal**

A força de cada driver varia por mês-calendário — mesmo com N≈83 por mês, o GBM não aprende isso sozinho com eficiência. Solução: **features explícitas índice × janela sazonal**. Como o teste cobre **todos os 12 meses de 2023 e 2024**, não há janela fora de escopo — todas as linhas da tabela são relevantes.

| Índice | Região de impacto | Janela de pico | Mecanismo |
|---|---|---|---|
| Niño 3.4 / ONI + **diversidade EP/CP** (Niño3 vs Niño4, ou C/E) | Sul/SESA, Amazônia | SON–DJF | Walker, subsidência; EP vs CP dão respostas distintas — **crítico para 2023 (El Niño forte)** |
| **TNA − TSA / AMM** (guardar TNA e TSA **separados** também — bacia toda quente importa) | Nordeste | FMA–MAM | Posição da ZCIT — **2023–24 teve Atlântico N anômalo recordista** |
| **ATL3 (Niño Atlântico)** | NEB oriental, Amazônia leste | MJJ–JJA | Modo zonal equatorial, ZCIT; limiar ~27 °C |
| **SAOD / SST Atlântico SW** | SACZ, SE/S Brasil–Uruguai | DJF | Dipolo subtropical do Atlântico Sul |
| **TIO (Índio tropical de bacia)** | Amazônia, NEB | DJF–MAM | Herança/capacitor do ENSO — mais relevante que DMI |
| SAM/AAO | Sul/Sudeste | DJF | Modo anular, frentes |
| IOD/DMI | Secundário | SON | Teleconexão fraca/controversa para SA |
| **PDO — como modulador, não ruído** | Amazônia, SESA | — | Feature de **interação** PDO×Niño3.4 (amplifica teleconexão ENSO→SA) |
| PMM | Secundário | — | Pacífico subtropical |
| SALLJ (V850 ~17°S), índice SACZ (OLR), **data de onset da monção** | Interior, transições | SON (onset) | Circulação interna da SA — crítico nos meses de transição |
| SST global como **campo** (PCs **detrended**) | Toda SA | — | Padrões que índices não capturam |
| Umidade do solo (ERA5-Land-T na simulação de disponibilidade) | SESA (hotspot GLACE), interior, transição seca→chuvosa | SON + meses de transição | Skill modesto e regional — concentrar em hotspots de acoplamento |
| **SSS (SMAP, pós-2010)** | Testável — série curta | — | Diferencial do vencedor de precipitação do Rodeo |

Interações explícitas mínimas: `Niño3.4×{SON,DJF}`, `(TNA−TSA)×{FMA,MAM}`, `ATL3×{MJJ,JJA}`, `SAOD×{DJF}`, `SAM×{DJF}`, `PDO×Niño3.4`.

**Decisão de dados:** usar **índices prontos** (NOAA CPC/PSD, JAMSTEC) sempre que existirem — economiza dias e elimina bugs de recomputação. Computar do zero só o que não existe pronto. Toda fonte externa entra no **manifesto de auditoria** (§6).

---

## 3. Decisões de dados — checklist do dia 0/1

Mecânica do Kaggle **resolvida** (aba Data/Overview): CSV `id,tp_mm_day`, RMSE absoluto, alvo = ERA5, teste = 2023+2024 num arquivo só, features de teste = mês M−1. Resta:

1. **Baixar os 13 arquivos oficiais (2,06 GB)** — via `kaggle competitions download -c previsao-climatica-de-precipitacao-sobre-a-america-do-sul` — e validar: dims (time, lat, lon), unidades mm/dia, padrão de NaN, `time_origem` vs. alvo, ordem exata do `sample_submission.csv`. Escrever o writer/validator de CSV contra o sample antes de qualquer modelo.
2. **Credenciais CDS + aceites de licença** (cada dataset C3S exige aceite individual; `cdsapi` ≥0.7 com URL nova). Disparar SEAS5 **imediatamente** se a via V3 for mantida: filas de horas–dias, fita MARS, requests grandes expiram — muitos requests pequenos (1 mês-início × todos os anos, subset SA, só precip + SST). **Kill-switch no fim do dia 2** (§4-V3).
3. **Fallback concreto de dinâmico:** NMME via IRI Data Library (OPeNDAP, sem licença, streaming, hindcast 1982+; CFSv2, GFDL, GEOS, CanSIPS) e PROCLIMA (CPTEC/FUNCEME).
4. **Fonte de SST:** ERA5 mistura HadISST2→OSTIA (~2007, quebra de homogeneidade). Decidir dia 1: **OISSTv2** (consistente, download trivial) ou SST do próprio ERA5 — e registrar a escolha.
5. **Log de disponibilidade por feature:** cada feature registra sua data de publicação real. Para o teste 2023–2024 tudo é histórico, mas o log continua obrigatório — é o que torna o pipeline auditável e defensável perante a banca.
6. **Auditoria desde o dia 0:** todo download externo com fonte/licença/URL/hash no manifesto; o código dos finalistas será inspecionado — tratar reprodutibilidade como requisito de ranking, não de relatório.
7. **Data de emissão/corte:** confirmar com a organização qual o último dado usável para prever t+1 (fim do mês t?) — condiciona a whitelist de start dates do V3.

---

## 4. Arquitetura

```
                 ┌──► [V0]  Climatologia + tendência (por fold!) ──► baseline E componente do modelo
                 ├──► [V0.5] Elastic-net linear multi-tarefa ──────┐
Dados (t e antes)┼──► [V1]  GBM global multi-task (workhorse) ─────┤
                 ├──► [V2]  Análogos PCs SST detrended + AutoKNN ──┼─► [Stacker: média /
                 └──► [V3*] Dinâmico debiased (SEAS5 → fallback    ┘    1/MSE / NNLS-ridge
                           NMME, kill-switch dia 2)                     sobre OOF]

ŷ = climatologia_fold + σ_pixel,mês × damp × ẑ     (ẑ = anomalia padronizada; clip ≥0)
V3* = via condicional — entra só se os dados estiverem em disco até o fim do dia 2
```

**Cortado do escopo (decisão explícita, não adiamento):** V4-CNN sobre campos, V5-ACE2 e augmentation CMIP6. Com ~10 dias de maratona, 3 submits/dia, LB-público armadilhoso e auditoria de código, o custo/benefício é claramente negativo — pipeline enxuto V0+V0.5+V1+V2(+V3 condicional) disputa o topo com consistência. A única ressurreição admissível, com núcleo estável e tempo sobrando, é o **MLP ensemble sobre as features oceânicas já existentes** (horas de custo, não dias).

**Mudanças conceituais mantidas da v3:**

- **Alvo = anomalia padronizada** `z = (y − clim)/σ_clim(pixel,mês)` — homocedástico, pooling coerente. Perda: Huber; opcional quantílico p10/p50/p90.
- **Damping:** encolhimento de anomalias rumo a zero, calibrado em OOF (global ou por macro-região) — skill quase grátis sob RMSE.
- **Ênfase nova:** com RMSE absoluto, o erro de `clim` entra sem atenuação no `ŷ`. A climatologia merece o mesmo rigor de modelagem que as vias: comparar no OOF janelas de referência (período cheio vs. 1991–2020 vs. últimos 30 anos), harmônicos e suavização espacial — é onde o RMSE se ganha ou se perde.

### V0 — Climatologia blindada (promovida a via crítica)

Refit **por fold** com harmônico + tendência; comparar variantes em OOF. É baseline, referência de MSSS **e** maior termo da saída final. Leakage aqui contamina tudo — o fold manifest precisa isolar anos do fold inclusive no cálculo de σ.

### V0.5 — Elastic-net linear multi-tarefa

Regressão linear local com seleção multi-tarefa sobre os mesmos preditores do GBM. Com N≈83/mês pode terminar #1–2 do stack e custa horas. Régua honesta para julgar se o GBM compra algo.

### V1 — GBM global multi-task (workhorse)

Um LightGBM/CatBoost para todos os pixels (lat, lon, elevação como features). Features:

- **Atmosfera do mês M−1 (oficial, disponível em TODO o teste):** t2m, cloud cover, SP, q850, rh850, T850, Z850, U850, V850 — pixel + vizinhança agregada (raio 2–3 px) + anomalias regionais (SACZ, SALLJ, ZCIT via gradiente)
- **Alvo/passado:** anomalia de `tp_ultima_obs` (dez/2022) × `lag_meses` como features (persistência decaindo), climatologia do próprio pixel como feature-base, anomalias padronizadas do histórico (média móvel 3m até dez/2022, lag-12), vizinhança agregada
- **Calendário:** sin/cos do mês **+ interações índice×janela da §2**
- **Oceano:** tabela completa da §2 com o **estado real observado até M−1** (não precisa de SST "prevista" — 2023–24 já é histórico; restrição causal do D-008) + primeiros PCs de SST **detrended**
- **Superfície/atmosfera extra (externo):** soil moisture ERA5-Land (hotspots + transição), TCWV, U/V 200, OLR, MSLP — anomalias regionais agregadas, com log de disponibilidade
- **Dinâmico (condicional ao V3):** previsão t+1 debiased + spread do ensemble como feature de incerteza — schema de features versionado para rodar sem a coluna se V3 cair
- Perda: Huber sobre z

### V2 — Análogos (duas variantes, ambas baratas)

- **V2a — Análogos de SST:** EOFs de SST **detrended**; domínio Pacífico + Atlântico tropical; distância em PCs padronizados; k≈5–10 com peso ∝ 1/(d+ε); **excluir anos do fold do pool**.
- **V2b — AutoKNN:** KNN na história da própria variável-alvo. Trivial de implementar.

### V3 — Dinâmico sazonal debiased (**condicional, com kill-switch**)

**Não é espinha dorsal — é upgrade condicional.** CDS/MARS tem filas de horas–dias e requests grandes expiram; numa maratona de 10 dias, 3–4 dias de download/decodificação de GRIB queimam o tempo de modelagem. Regra dura: **se o hindcast não estiver baixado e decodificado até o fim do dia 2, acionar NMME via IRI OPeNDAP; se falhar, dropar a via** e concentrar em V0.5+V1+V2. Não se negocia o relógio com a fila do CDS.

Se fluir:

- **Datas de inicialização — whitelist por sistema** (lead real ~4–5 semanas para "mês t+1"): SEAS5/DWD/CMCC/MF inicializam dia 1; GloSea6 diário (hindcast 1/9/17/25); ECCC/JMA no fim do mês anterior; CFSv2 a cada ~5 dias. **Armadilha crítica:** run inicializado *dentro* de t+1 embute ~2 semanas de observação do mês-alvo → leakage. Data de emissão formal conforme §3.7.
- **Hindcast↔realtime — agora certeza, não risco:** o teste (2023–24) é real-time puro; membros diferem (SEAS5 25 hindcast vs 51 realtime) e climatologias de referência mudam → calibração treinada em hindcast deve ser conservadora para transferir. Pareamento hindcast↔sistema idêntico continua requisito.
- **Debias paramétrico, não QM livre:** correção de **média+variância** (ou gamma–gamma), pooling ±1 mês se necessário, **refitada por fold**. Com N≈83/mês o QM livre ainda tem erro de cauda alto demais.

### Stacker

NNLS ou ridge encolhido para pesos iguais sobre anomalias padronizadas OOF. Benchmarks obrigatórios: **média simples** e pesos ∝1/MSE (forecast combination puzzle: a média frequentemente ganha). No máximo **2 conjuntos sazonais de pesos** (estação úmida/seca). Reportar estabilidade dos pesos entre folds + skill OOF por via.

---

## 5. Validação — protocolo anti-leakage e anti-armadilha-2023

- **CV temporal:** LOYO com **embargo ±1 ano** (features lag-12 embutem estado do ano-teste). **Nunca shuffle.** Holdout interno congelado = últimos ~5 anos do treino (≈2018–2022), intocado até a seleção final.
- **Refit por fold:** climatologia e σ, EOFs de SST, debias do V3, pool de análogos, scalers — tudo recalculado sem os anos do fold. A climatologia é o item mais sensível agora que a métrica é absoluta.
- **Painel de métricas:** RMSE absoluto (a oficial) + ACC temporal por pixel, MSSS vs. climatologia e persistência, taxa de acerto de tercil, **breakdown por macro-região (Amazônia, NEB, Centro-Oeste, Sul/SESA), mês-calendário e fase ENSO** — a assimetria convectiva×dinâmica da Dra. Marília vira diagnóstico obrigatório — e **bootstrap por bloco-ano** para IC de diferenças entre vias.
- **Anti-armadilha-2023 (regra explícita):** o LB público mede um ano de transição ENSO extrema; o privado mede outro regime. **CV temporal decide modelo, hiperparâmetros e pesos de stacker; os 3 submits/dia servem para confirmar CV e testar infra, nunca para hill-climbing.** Submissões finais escolhidas por OOF + diversidade estrutural entre vias. Registrar no log: submit ↔ git hash ↔ OOF ↔ score LB.
- **Datas de emissão:** whitelist de start dates (§4-V3); log de disponibilidade por feature (§3.5).

---

## 6. Engenharia

- **OOF prediction store:** cada via grava parquet `(data, lat, lon, pred, fold, via)` versionado — sem isso o stacking não existe.
- **Manifesto de folds** fixo em disco + **log de submits** (submit ↔ git hash ↔ OOF ↔ LB).
- **Writer/validator do CSV de submissão** (~943k linhas/ano): ID exato `{ano}_{mes}_{latitude}_{longitude}`, formatação de float de lat/lon **igual** à do sample_submission, contagem de linhas, NaN check, clip ≥0. Validar antes do primeiro submit — o V0 do dia 1 existe em grande parte para testar esse caminho de ponta a ponta.
- **Pronto para auditoria (requisito de ranking):** repo organizado, README de reprodução ponta-a-ponta, `environment.yml` pinado, seeds fixas, **manifesto de dados externos** (fonte, licença, URL, data de download, hash) e downloader C3S com manifesto/retry — se usarmos dados externos, o pipeline de download+prep precisa ser reproduzível e documentado, como frisado pelo Jerônimo.
- Máscara terra/oceano + política de fill de SST; hashes dos zarr intermediários.
- Seeds/bagging temporal no GBM; suavização espacial opcional na saída.
- GPU Kaggle (30 h/sem, ≤12 h/sessão) existe, mas **não reabre** os moonshots — serve para batch scoring/GBM-GPU se necessário.
- **Execução pesada migra para Kaggle kernels** (16/09): laptop local tem ~2 GB RAM livre; kernel script privado (`thisux1/worcap-v1`) + dataset privado `thisux1/worcap-assets` (código `worcap/` + parquets de índices/features) com `competition_sources` anexado. `config.py` aceita env vars (`WORCAP_DATA/EXT/SUB/OOF`) — mesmo código roda local e no Kaggle. Outputs baixados via API e submetidos por `kaggle competitions submit` local.
- CHIRPS/MSWEP: só como parágrafo de diagnóstico físico no relatório (viés do ERA5 vs. observações), se sobrar uma tarde. Sem papel corretivo — o alvo é o ERA5.

---

## 7. Cronograma revisado (restam ~7 dias — competição já está aberta)

| Dia | Entrega |
|---|---|
| 1 (hoje) | `kaggle` CLI + download dos 13 arquivos; **CSV writer/validator** contra sample_submission; índices NOAA prontos; pipeline xarray + climatologia por fold → **submit V0** (testa schema end-to-end); fold manifest + OOF store + manifesto de auditoria; spike lilio (2 h); decidir V3: disparar SEAS5/CDS **ou** ir direto a NMME/IRI |
| 2 | V0.5 elastic-net → submit; V1 GBM v0 (9 vars oficiais M−1 + climatologia + índices) → submit; **kill-switch V3**: dados em disco? senão NMME/IRI ou drop |
| 3 | V1 full (interações índice×janela, teleconexões, solo); V2a+V2b; V3-lite só se dados já disponíveis |
| 4 | Stacker v1 + benchmarks (média/1-MSE/NNLS); damping; painel de métricas por macro-região/mês/fase ENSO; spike FuXi-S2S só se V3 morto e tempo sobrando |
| 5 | Tuning leve; seeds/bagging; blindagem da climatologia (variantes em OOF); documentação p/ auditoria |
| 6 | Ensemble final por OOF + diversidade; holdout congelado confirma; relatório + README de reprodução |
| 7 | **Submits finais** (2 escolhidos por OOF+diversidade); buffer p/ imprevistos |

**Divisão de trabalho (4 pessoas):** P1 = pipeline CV/OOF/submissão + V1 (peça crítica — ML mais forte); P2 = dados externos + V3/NMME + manifesto de auditoria (maior risco, começa dia 0); P3 = V2 + features oceano + relatório/banca; P4 = V0/V0.5 + climatologia/damping + painel de métricas + validação de CSV.

**Política de submits:** máx. 3/dia, alocados para confirmação de CV — não para exploração. Guardar slots para os dias finais.

---

## 8. Perguntas abertas para o time

Resolvidas pela aba Data: schema do CSV, unidades (mm/dia), grade, período de teste, emissão (features de M−1). Restam (detalhadas em DECISIONS.md — P-001..P-007):

1. **Tamanho do time:** vídeo diz ≤4, LinkedIn oficial diz ≤3 — confirmar na aba Rules/Discord.
2. **Dado externo do próprio mês-alvo** (SST observada de jan/2023 p/ prever jan/2023): permitido? Default = não usar (causal ≤M−1).
3. **Formato de citação exigido** para dados externos + escopo da auditoria de código (aba Rules).
4. **Recursão de previsão** como feature para 2024: avaliar em OOF (risco de propagação de erro) — decisão interna, não de regra.

---

## Apêndice — Fontes consultadas

- **Regras e formato confirmados:** vídeos de organização do hackathon (Jerônimo; Carlos) — métrica RMSE absoluto, alvo ERA5, treino 1940–2022, LB público 2023 / privado 2024, CSV `ID={ano}_{mes}_{lat}_{lon}`, 3 submits/dia, times ≤4, auditoria de código; palestra da Dra. Marília — memória atmosférica ~10–14 dias vs. oceano meses, convecção subgrade na Amazônia/Centro-Oeste, resposta frontal/ENSO no Sul, ERA5 como reanálise acoplada
- Subseasonal Climate Forecast Rodeo: Hwang et al. (KDD'19, arXiv 1809.07394) — vencedor: MultiLLR + AutoKNN + dinâmico debiased; precipitação: ensemble de MLPs sobre features oceânicas incl. SSS (Team Salient)
- Pinheiro & Ouarda, "An interpretable machine learning model for seasonal precipitation forecasting" (Comms. Earth & Env. 2025) — TelNet (avaliado no Ceará, escala sazonal)
- Pinheiro & Ouarda, "Enhancing machine learning-based seasonal precipitation forecasting using CMIP6 simulations" (Atmos. Research 2025)
- Domingos et al., arXiv 2512.13910 (avaliação em 2019 apenas)
- Zeng et al., "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
- ACE2 (npj CAS 2025; `ai2cm/ace`) — cortado do escopo
- C3S seasonal forecast multi-system (CDS): hindcasts heterogêneos (SEAS5 1981–2016; CFSv2 1993–2016; JMA 1993–2020; ECCC 1980–2023), membros hindcast≠realtime — teste 2023–24 é real-time puro
- NMME via IRI Data Library (OPeNDAP); PROCLIMA (CPTEC/FUNCEME); índices prontos NOAA CPC/PSD, JAMSTEC
- Teleconexões SA: TNA−TSA/ZCIT/NEB (Nobre & Shukla 1996; Giannini; Servain); ATL3/modo zonal atlântico; SAOD/SACZ (Doyle & Barros 2002); PDO×ENSO (Andreoli & Kayano 2005); capacitor TIO; hotspots GLACE/SESA
- CHIRPS/MSWEP: apenas diagnóstico de viés do ERA5 (AGU GHH 2024; Atmosphere 2025 Bolívia)
