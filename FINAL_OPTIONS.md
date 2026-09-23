# Opções de seleção final — WORCAP (decisão até 22/09)

*Documento para avaliação por agentes. Todos os números verificados no LB ou em kernels OOF.*

## Mecânica da decisão

- Score final = **RMSE privado** = linhas-alvo de **2024** apenas (12 meses × 78.561 px).
- Público (2023) é irrelevante ao prêmio — só ranking de exibição.
- Kaggle: marcamos **2 submissões** do histórico; o sistema fica com a **melhor no privado** (min-of-2).
- Desempate: submissão mais antiga vence → candidatos devem já estar submetidos.
- Regime 2024: decay de El Niño forte (jan–mai quentes por n34 origem ≥0.8; jun–dez neutro).

## Evidência-chave

1. **Offset ENSO confirmado no público**: A = blend65/35 + β0.25/thr0.8 → **1.68586** (−0.0123 vs baseline 1.69816). Dose-resposta β0.15→−0.0085, β0.25→−0.0123.
2. **Offset em OOF por ano** (gap1, thr0.8): 1983 −0.037, 1992 −0.014, 1997 −0.015, 1998 −0.018, 2015 −0.011; 1988 +0.013 (La Niña), 2005/2010 ~+0.003, **2016 +0.019** (o decay-análogo mais próximo de 2024!).
3. **γ>1 global morto no LB** (+0.0026); por fase OOF: ajuda em meses quentes de 1983/92/97/98 (−0.010~−0.018) mas **2016 +0.058** e 2015 +0.015 → γ definitivamente fora.
4. **v2 (ridge por pixel)**: −0.022 OOF mas +0.009 LB-2023 — pior transferência documentada.
5. **2016 é o dissidente central**: evento 2015-16 = forte + flavor CP/costero — o mais parecido com 2023-24 no eixo Pacífico. Rejeita tanto γ>1 (+0.058) quanto offset (+0.019) porque a resposta de chuva foi muda/deslocada vs a canônica. *(Correção pós-avaliação: o TNA de 2016 foi +0.2, banal — NÃO era Atlântico quente; o discriminador Atlântico separa 2024 de 2016.)*
6. **Contra-evidência revisada**: 2023-24 teve Atlântico recorde absoluto (TNA jan-mai/24 = +1.2~1.8, máximo da série 1948-2026). MAS: (a) literatura 2025 mostra que o Atlântico/Índico recordes **suprimiram** a teleconexão Pacífica (~1/3 dos super-eventos) → assinatura 2024 é muda tipo-2016, não amplificada; (b) 2010, o outro decay com TNA recorde (+1.25), teve ganho de offset ≈ 0 → "Atlântico amplifica offset" é refutado pelo próprio painel. Ganhos grandes do offset só ocorreram com Atlântico frio (1983, 1992, 1998).
7. **λ2=30**: −0.002 borda de grid, 4 anos só → não adotado. Winsorize/x2std/pooling/recursão/damping/NMME: mortos.

## Inventário submetido (selecionável)

| tag | público | composição 2024 |
|---|---|---|
| **A** `blend_v2_65_enso_b25` | **1.68586** | blend + offset β0.25 nos 5 meses quentes |
| **A33** `splice_A_b33_24` | 1.68586 | 2023 = A; 2024 = blend + offset **β0.33** |
| `splice_A_ridgeEnso24` (S1) | 1.68586 | ridge_all + offset β0.25 (sem v2) |
| `splice_A_gwarm24` (S2, ordem desviada) | 1.68586 | blend + offset + γ1.10 nos 5 meses quentes |
| `blend_v2_65_enso_b15` (D) | 1.68963 | blend + offset β0.15 |
| `blend_v2_65_enso_b25_g110` (C) | 1.68847 | blend + offset + γ1.10 global |
| `blend_ridge_v2_65` | 1.69816 | blend puro (sem offset) |
| `v05_ridge_all` | 1.69966 | stage-1 puro (sem v2, sem offset) |
| `blend_v2_65_g110` | (gerado, não submetido) | — |

Gerável sem refit: splices arbitrários `[2023:X ‖ 2024:Y]`, variantes β, ridge_all+offset, blend65 sem offset.

## Opções de final2 (final1 = A é consenso)

### Opção R — `v05_ridge_all` (cobertura máxima)
- Cobre: offset-fail-2024 (cenário 2016) **e** v2-fail-2024.
- Argumento pró: único candidato sem nenhum lever contestado; já selecionável, mais antigo.
- Argumento contra: se offset transfere (P~60%?), perde para A e S1; público 1.69966 (irrelevante).

### Opção S1 — `splice_A_ridgeEnso24` (hedge v2, mantém offset)
- Cobre: v2-fail-2024. Mantém offset (lever confirmado) em jan-mai/24.
- Argumento pró: mantém upside ENSO; diversifica só na dimensão contestada (v2).
- Argumento contra: **não cobre o modo de falha evidenciado por 2016** (offset-fail em decay).

### Opção P — `blend_ridge_v2_65` (offset-free, mantém v2)
- Cobre: offset-fail-2024. Mantém v2 (upside OOF gap2 diz w→0.4, mas LB-2023 diz v2 +0.009).
- **Custo: ZERO** — correção pós-avaliação: o lado-2024 do splice `[A‖blend65]` é idêntico a `blend_ridge_v2_65`, já submetido (1.69816). O splice seria cosmética de placar público.
- Idem H: lado-2024 de `[A‖β0.15]` ≡ `blend_v2_65_enso_b15` = D, já submetido.

### Opção H — splice novo `[A-2023 ‖ A-β0.15-2024]` (meia-dose privada)
- Cobre parcialmente offset-fail (metade do dano se falha, metade do ganho se funciona).
- Hedge "suave" — menos cobertura, mais EV em mundos bons.

### Opção C — combinações (ex.: [2024: ridge_all+offset+β0.35])
- Compostas: só vencem se TODAS as mudanças ajudarem → dominadas sob min-of-2.

## Vereditos dos avaliadores (6 agentes, 2 rodadas)

**Rodada 1 (final2, final1=A travada):** EV-max→R; minimax→R (P runner-up); advogado→P. *Posteriormente superada pela rodada 2.*

**Rodada 2 (dose × hedge acoplados):**

| agente | dose | hedge | argumento central |
|---|---|---|---|
| EV-max | **A33** | **P** | min() torna o bump opção de upside quase grátis; v2 decorrelaciona mesmo quando falha |
| minimax-regret | **A33** | **P** | max-regret 6 vs 8 (A25P) / 8 (A25R) / 10 (A33R); Δ(β) quadrático → overshoot band estreita (α∈0.12-0.17, ≤0.006) |
| advogado | A25 (ou β0.4) | R | objeção de processo (dose contaminada pelo slope LB); v2 falha em regime extremo |

## DECISÃO (dia 22)

**final1 = `splice_A_b33_24` · final2 = `blend_ridge_v2_65`** — par {A33, P}.

- Dose justificada por ótimo OOF-análogo dos anos-decay (β*≈0.30-0.60; tabela sessão 11), NÃO pelo slope do LB. Alegação "Atlântico amplifica" descartada (2010 refuta).
- Hedge P > R: cobre o mundo offset-falha mantendo v2; R só vence na falha-composta (exige P(v2-falha|off-falha)>0.62, contradito por gap2). Margem ~0.001-0.005 — quase cara-ou-coroa, registrado.
- **Auto-seleção do Kaggle pegaria {A, S1} (ambos com offset) → seleção manual obrigatória.**
