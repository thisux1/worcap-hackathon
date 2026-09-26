# Post-mortem — WORCAP 2026

Fechei em ~1.83875 no privado (par `splice_A_b33_24` + `blend_ridge_v2_65`),
fora do top-10 (corte em 1.80337). O vencedor fechou em 1.57992. Depois do
encerramento ele publicou o código e a organização esclareceu no canal oficial
o que era permitido. Escrevi isso pra registrar o que eu faria diferente.

## O que o vencedor fez

O modelo dele não é estatístico puro. É uma previsão dinâmica com correção
estatística em cima:

- A base P é um NNLS combinando EOFs do estado atmosférico, regressão por
  célula, climatologia e a anomalia do CFSv2, com correção do ensemble GEFS.
  Truncado em zero no final.
- O SEAS5 entra como variável central, como anomalia contra a climatologia do
  hindcast 1993-2016 do próprio sistema (SEAS5 e SEAS5.1 têm climatologias
  separadas; misturar cria sinal falso).
- Os corretores aprendem o resíduo `r = Y − P`: LightGBM, linear e um MOS com
  ridge na vizinhança 3×3 (modelo global acerta o padrão mas desloca 1-2°).
  Ensemble com pesos por região e estação, encolhidos pra média.
- A disciplina era a mesma que a minha: blocos de ano, holdout 2020-22,
  hipóteses pré-registradas, público nunca usado pra escolher modelo. Vale
  notar: uma rede neural no resíduo piorou o resultado dele.

O gap não veio de tuning. Veio de conteúdo: a previsão dinâmica carrega a
evolução atmosférica dentro do mês-alvo, e nenhuma feature estatística de T−1
recupera isso. É um teto do tipo de informação que eu alimentei o modelo, não
do modelo em si.

## Onde eu errei

1. Cortei a via dinâmica (V3) no dia 2 por medo da fila do CDS/MARS. Depois
   acabei baixando NMME mesmo assim, mas usei só como feature marginal numa
   tabela de 108 colunas. Ele fez disso o centro do modelo. A fila era
   real, mas era custo fixo: era só disparar o download em background no dia 0
   enquanto eu construía o baseline.
2. Li a regra de causalidade mais estrita do que ela era. Assumi "features ≤
   fim do mês M−1" (D-008). O critério oficial era não usar informação
   posterior ao mês previsto. Um SEAS5 inicializado no dia 1 do mês-alvo não
   observa o mês, ele prevê. Podia ter usado lead-0 e não usei. Podia ter
   perguntado no canal no dia 0 e não perguntei.
3. Gastei submits e dias em micro-otimização (γ, damping, blends, sweep de λ,
   coisas de 0.002 a 0.01) com a alavanca grande parada em `data_ext/`.
4. O stack linear fecha em ~1.84 no privado. Honesto e explicável, mas 2024
   precisava de informação que reanálise de T−1 simplesmente não tem.

## Se fosse de novo

- Dia 0: `baixar_seas5`/`nmme` rodando em background antes de qualquer modelo.
- Dinâmica como backbone (NNLS com SEAS5 + CFSv2 + GEFS), não como feature.
- MOS por pixel com vizinhança 3×3. É a ideia mais barata do stack dele e
  provavelmente a mais valiosa.
- Corretor no resíduo em vez de prever a anomalia direto.

## O que eu manteria

- O protocolo anti-leakage (LOYO + embargo, refit por fold). Passou na
  verificação de código sem nenhuma pendência.
- O hedge nas finais. O β0.33 overshooteou no privado (1.84017) e o blend sem
  offset segurou o par em 1.83875. A arquitetura barbell fez o que prometia.
- O offset ENSO funcionou no público (−0.0123) e o modo de falha apareceu no
  OOF antes de custar nada (2016 → hedge).
- O DECISIONS como log datado. Sem ele eu não conseguiria escrever esse arquivo
  nem defender nenhuma escolha na auditoria.

## Resumo

A validação me deixou no pelotão honesto, mas quem subiu no pódio respondeu
antes "qual fonte de dados carrega o estado real do mês-alvo?" e só depois
"qual modelo?". Eu fiz as duas perguntas na ordem errada: otimizei o modelo em
cima de um conjunto de features que não continha a resposta.
