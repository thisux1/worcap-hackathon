# Post-mortem — WORCAP 2026

Resultado: ~1.83875 privado (par `splice_A_b33_24` + `blend_ridge_v2_65`), fora do
top-10 (corte 1.80337). Vencedor: 1.57992 — gap de ~0.26 RMSE. Este arquivo
registra onde a diferença nasceu e o que faríamos diferente. Escrito depois do
encerramento, com o repositório do vencedor (`github.com/Brun0Simoes/Athon`)
publicado e o debate do canal #duvidas já resolvido pela organização.

## O que venceu

A solução campeã não é estatística pura: é estatística-dinâmica.

- **Base dinâmica P**: NNLS combinando EOFs do estado atmosférico, regressão por
  célula, climatologia e anomalia do CFSv2; correção com ensemble GEFS (NOAA);
  truncado em zero.
- **SEAS5 como variável central**: anomalia vs a climatologia de hindcast
  1993-2016 do próprio sistema (SEAS5 e SEAS5.1 têm climatologias separadas).
- **Corretores aprendem o resíduo** `r = Y − P`: LightGBM + linear + MOS
  (ridge com vizinhança 3×3, porque modelo global acerta o padrão mas desloca
  1-2°). Ensemble com pesos por região × estação, encolhidos pra média.
- Disciplina: blocos de anos, holdout 2020-22, hipóteses pré-registradas,
  público nunca usado pra escolher modelo. Rede neural no resíduo *piorou* —
  modelos simples e robustos ganharam.

O ganho vem de conteúdo de informação: previsão dinâmica carrega a evolução
atmosférica **dentro** do mês-alvo. Nenhuma feature estatística de T−1 recupera
isso — é um teto estrutural, não uma questão de tuning.

## Onde perdemos (decisões com custo real)

1. **V3 dinâmica morta no dia 2** (D-003, D-015). Cortamos a via SEAS5/NMME por
   medo de fila do CDS/MARS. Depois baixamos NMME mesmo assim — e usamos só como
   feature marginal. O vencedor fez dela a espinha dorsal. O medo da fila se
   concretizou parcialmente, mas downloads deveriam ter começado no dia 0 em
   background enquanto o baseline estatístico era construído.
2. **Causalidade mais estrita que a regra** (D-008). Autoimpusemos "features ≤
   fim do mês M−1". O critério oficial era ausência de informação *posterior ao
   mês previsto* — previsão dinâmica inicializada no dia 1 do mês-alvo (lead-0)
   é legal: ela não observa o mês, prevê. A organização confirmou isso no canal.
   Nossa leitura conservadora excluiu justamente o sinal decisivo.
3. **Sequência de experimentos**: gastamos submits e dias em micro-otimizações
   (γ, damping, blends, λ-sweep — ganhos de ~0.002-0.01) enquanto a alavanca
   macro (sinal dinâmico) ficou parada em `data_ext/`.
4. **Teto do stack linear**: ridge + pixel-ridge + compósito ENSO fecha em
   ~1.84 privado. É honesto, auditável e explicável — mas o regime de 2024
   precisava de informação que reanálise de T−1 não contém.

## O que faríamos diferente

- Dia 0: disparar `baixar_seas5`/`nmme` em background antes de qualquer modelo.
  Fila de download não é razão para cortar a via — é custo fixo.
- Tratar previsão dinâmica lead-0/lead-1 como **backbone** (NNLS/multi-modelo:
  SEAS5 + CFSv2 + GEFS), não como feature em tabela de 108 colunas.
- MOS por pixel com vizinhança 3×3 (ridge) — o insight de "padrão certo no lugar
  errado" é o mais barato e provavelmente mais valioso do stack dele.
- Corretor sobre resíduo em vez de previsão direta da anomalia.
- Ler a regra de causalidade como "conteúdo informacional", não "data de
  publicação" — e confirmar com a organização **no dia 0**, não assumir.
- Mesma disciplina que já tínhamos: blocos/LOYO, holdout intocado, hipótese
  pré-registrada, público só como confirmação.

## O que deu certo (manter)

- Protocolo anti-leakage (LOYO + embargo, refit por fold) — zero retrabalho de
  auditoria; código passou na verificação.
- Hedge estrutural nas finais ({A33, P}): quando o offset overshooteou no
  privado (β0.33 → 1.84017), o blend sem offset segurou o piso (1.83875).
- Documentação como artefato do processo (DECISIONS/FINAL_OPTIONS/ADR) —
  possibilitou reconstruir o porquê de cada escolha, inclusive a errada.
- O offset ENSO-condicional funcionou de verdade no público (−0.0123) e o modo
  de falha foi identificado antes de custar (2016 → hedge).

## Lição central

Validação rigorosa te coloca no pelotão honesto; **conteúdo de informação**
decide o pódio. Em problema de previsão, a primeira pergunta é "qual fonte de
dados carrega o estado real do alvo?" — não "qual modelo". As três primeiras
posições provavelmente todas responderam a primeira pergunta com previsão
dinâmica. A gente respondeu a segunda com ridge.
