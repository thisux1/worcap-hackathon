# Opções de seleção final — WORCAP (decisão até 22/09)

Documento montado para avaliação independente por agentes. Números todos
verificados no LB ou em kernels OOF.

## Mecânica

- O score final é o RMSE privado, calculado só nas linhas de 2024 (12 meses × 78.561 px).
- O público (2023) não vale prêmio; serve de ranking de exibição.
- Marcamos 2 submissões do histórico e o Kaggle fica com a melhor no privado (min-of-2).
- Desempate favorece a submissão mais antiga, então candidatos precisam já estar submetidos.
- Regime de 2024: decay de El Niño forte. Jan–mai têm n34 de origem ≥ 0.8 (quentes); jun–dez neutro.

## Evidência

1. Offset ENSO confirmado no público: blend65/35 + β0.25/thr0.8 deu 1.68586, −0.0123 contra o baseline 1.69816. Dose-resposta: β0.15 → −0.0085, β0.25 → −0.0123.
2. Offset em OOF por ano (gap1, thr0.8): 1983 −0.037, 1992 −0.014, 1997 −0.015, 1998 −0.018, 2015 −0.011; 1988 +0.013 (La Niña), 2005/2010 ~+0.003, 2016 +0.019 (decay mais próximo de 2024).
3. γ>1 global morreu no LB (+0.0026). Por fase no OOF ajuda em meses quentes de 1983/92/97/98 (−0.010~−0.018), mas 2016 dá +0.058 e 2015 +0.015. Descartado.
4. v2 (ridge por pixel): −0.022 no OOF, +0.009 no LB-2023. Pior transferência documentada.
5. 2016 é o dissidente central: evento 2015-16 forte, com sabor CP/costero, o mais parecido com 2023-24 no eixo Pacífico. Rejeita γ>1 (+0.058) e offset (+0.019) porque a resposta de chuva saiu muda/deslocada do padrão canônico. Correção posterior: o TNA de 2016 foi ~+0.2, banal. Não era Atlântico quente; o discriminador Atlântico separa 2024 de 2016.
6. Contra-evidência revisada: 2023-24 teve Atlântico recorde (TNA jan–mai/24 = +1.2~1.8, máximo da série 1948–2026). Mas a literatura de 2025 mostra que o Atlântico/Índico recordes suprimiram a teleconexão do Pacífico (~1/3 dos super-eventos), ou seja a assinatura de 2024 é muda, tipo 2016, não amplificada. E 2010, o outro decay com TNA recorde (+1.25), teve ganho de offset ~0. A hipótese "Atlântico amplifica o offset" é refutada pelo próprio painel: os ganhos grandes vieram de anos com Atlântico frio (1983, 1992, 1998).
7. λ2=30: −0.002 na borda de grid, 4 anos só, não adotado. Winsorize/x2std/pooling/recursão/damping/NMME: mortos.
8. Compósito bidimensional (n34 × tna/atl3/nino12/ep_cp, kernel enso2): nenhuma das 8 variantes conserta 2016; melhor agregado marginal +0.0007, barulho. Em 2024 seria no-op de qualquer forma (células vazias caem no fallback). Experimento fechado.

## Inventário submetido

| tag | público | composição 2024 |
|---|---|---|
| A `blend_v2_65_enso_b25` | 1.68586 | blend + offset β0.25 nos 5 meses quentes |
| A33 `splice_A_b33_24` | 1.68586 | 2023 = A; 2024 = blend + offset β0.33 |
| `splice_A_ridgeEnso24` (S1) | 1.68586 | ridge_all + offset β0.25 (sem v2) |
| `splice_A_gwarm24` (S2) | 1.68586 | blend + offset + γ1.10 nos 5 meses quentes |
| `blend_v2_65_enso_b15` (D) | 1.68963 | blend + offset β0.15 |
| `blend_v2_65_enso_b25_g110` (C) | 1.68847 | blend + offset + γ1.10 global |
| `blend_ridge_v2_65` (P) | 1.69816 | blend puro, sem offset |
| `v05_ridge_all` (R) | 1.69966 | stage-1 puro, sem v2 e sem offset |
| `era5_truth_2023` | 0.00002 | probe ilustrativo, não elegível (verdade ERA5 nas linhas-2023; ver DECISIONS) |

## Opções avaliadas para final2 (final1 era A na época)

R (`v05_ridge_all`): cobre offset-fail e v2-fail juntos; único sem nenhum lever
contestado; já submetido e o mais antigo.

S1: cobre v2-fail mantendo o offset em 2024. Não cobre o modo de falha que 2016
mostrou (offset falhando em decay).

P (`blend_ridge_v2_65`): cobre offset-fail mantendo v2. O lado-2024 do splice
[A‖blend65] é idêntico ao arquivo já submetido, então custa zero. Mesmo vale
para H: o lado-2024 de [A‖β0.15] é o arquivo D já submetido.

H (`blend_v2_65_enso_b15` como hedge de meia-dose): cobre metade do dano se o
offset falha, metade do ganho se funciona.

S2: soma γ na mesma célula do offset. Aposta correlata, não hedge.

## Vereditos dos avaliadores

Rodada 1 (final2, final1=A travado): EV-max votou R; minimax votou R com P de
runner-up; advogado votou P. Superada pela rodada 2, que avaliou dose e hedge
juntos.

Rodada 2:

| agente | dose | hedge | argumento |
|---|---|---|---|
| EV-max | A33 | P | o min() transforma o bump em opção de upside quase grátis; v2 decorrelaciona mesmo quando falha sozinho |
| minimax-regret | A33 | P | max-regret 6 contra 8 (A25P), 8 (A25R), 10 (A33R); Δ(β) é quadrático, então a banda de overshoot é estreita (α∈0.12–0.17, custo ≤0.006) |
| advogado | A25 (ou β0.4) | R | objeção de processo: a dose 0.33 nasceu contaminada pelo slope do LB; v2 falha em regime extremo |

## Decisão (dia 22)

final1 = `splice_A_b33_24`, final2 = `blend_ridge_v2_65`.

A dose 0.33 vem do ótimo OOF dos anos-decay (β*≈0.30–0.60 na tabela da sessão 11),
não do slope do LB. O hedge é P e não R porque cobre o mundo offset-falha sem
abrir mão do v2; R só ganharia na falha composta, que exigiria P(v2-falha | off-falha)
> 0.62, contradito pelo gap2. A margem entre os dois é ~0.001–0.005, quase
cara-ou-coroa; registrado aqui por honestidade.

Atenção: a auto-seleção do Kaggle pegaria {A, S1}, ambos com offset. A seleção
manual é obrigatória.
