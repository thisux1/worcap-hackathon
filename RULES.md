# Regras da Competição — WORCAP 2026 (transcrição da aba Rules + esclarecimentos oficiais)

Fonte: aba "Rules" da competição (acesso restrito a inscritos) + mensagens do organizador
(geronimo.lemos) no canal oficial, transcritas em 18/09/2026.

## Regras específicas (COMPETITION-SPECIFIC TERMS)

- **Título:** WORCAP 2026 — Previsão Mensal de Precipitação na América do Sul.
- **Organizador:** PPG INPE — Workshop de Computação Aplicada (WORCAP 2026).
- **Premiação:** 1º–3º: certificado + reconhecimento no ranking oficial (sem prêmio em dinheiro).
- **Tamanho máximo da equipe: 4 integrantes.** Fusões permitidas dentro do limite de envios acumulados.
- **Submissões: máximo 5 por dia** (correção oficial: anunciado inicialmente como 3).
- **Dados da competição:** uso restrito a fins não comerciais (competição, fóruns, pesquisa acadêmica).
- **Licença do código:** a submissão vencedora e seu código-fonte devem ser licenciados sob licença
  aprovada pela Open Source Initiative, **sem restrição de uso comercial**.
- **Dados externos:** permitidos, desde que (a) de domínio público e igualmente acessíveis a todos
  os participantes sem custo, ou (b) atendam aos critérios de razoabilidade da Seção 2.6.b.
- **Obrigações do vencedor:** entregar código-fonte do modelo final (treinamento + inferência),
  documentação associada e descrição do ambiente computacional; o código deve ser capaz de gerar
  a submissão vencedora.

## Foundational Rules (pontos relevantes)

- Uma conta Kaggle por participante; proibido compartilhamento privado de código entre equipes
  (compartilhamento público no fórum da competição é permitido e licenciado OSI).
- Código open-source usado no modelo deve ter licença OSI sem limite de uso comercial.
- Ranking final = Private Leaderboard; desempate premia a submissão **mais antiga**.
- Submissões não podem usar rotulação manual/previsão humana dos dados de teste.
- Vencedor potencial deve responder à notificação em 1 semana; documentos de aceite em 2 semanas.

## Esclarecimentos do organizador (canal oficial)

**Anúncio de abertura (19h11):**
> "dado o estado atmosférico observado de um mês, prever a precipitação média do mês seguinte,
> em mm/dia, em cada ponto da grade sobre a América do Sul. Vocês recebem dez variáveis do ERA5
> em resolução de 0,25°, de 1940 a 2022, para treinar. A avaliação cobre 24 meses: 2023 alimenta
> o placar público durante a semana e 2024 decide o resultado final. Métrica única: RMSE sobre a
> chuva absoluta. Equipes de até 4 pessoas, 3 envios por dia. Dados externos são liberados, desde
> que públicos e declarados."

**Correção de limite (12h12):**
> "haviámos informado, de forma equivocada, que o limite de envios diário seria 3. Na verdade,
> é possível realizar 5 submissões por dia."

**Regra de causalidade — esclarecimento explícito (19h15):**
> "para prever um mês T, vale qualquer dado que em tese estaria disponível até o fim do mês T−1.
> Nada posterior a isso. Na prática, para prever fevereiro de 2023 você pode usar tudo até janeiro
> daquele ano, como os campos que já vêm no teste_features.nc, dados externos em escala diária ou
> semanal, índices climáticos, informação de fora do domínio. Daí decorre uma consequência que vale
> explicitar: os campos de janeiro de 2023, que vêm no arquivo como preditores de fevereiro, não
> podem ser usados para estimar a chuva de janeiro de 2023. Reorganizar o arquivo para diagnosticar
> o próprio mês não é previsão."

## Consequências operacionais para o nosso pipeline

1. **Corte causal confirmado:** features para o alvo T limitadas a dados até fim de T−1. Nosso
   pipeline já indexa por `time_origem = T−1` (ver `worcap/features.py`) — compliant por construção.
   Dados externos do próprio mês-alvo T são **proibidos** (resolve P-007: nosso default D-008 era o
   correto).
2. **Dados externos fora do domínio** explicitamente liberados — legitima ERSSTv5 (Pacífico/Atlântico
   globais), ERA5 MSLP Southern Ocean (fora da grade oficial), NMME/SEAS5 via C3S.
3. **Features diárias/semanais externas** são legais até fim de T−1 (não explorado — mensal basta).
4. **NMME lead-1:** forecast com init no mês T−1 prevendo T está disponível ~dia 8–10 de T−1
   (real-time), dentro da janela — legal pela regra "em tese disponível até fim de T−1".
5. **Licença:** repo licenciado MIT (LICENSE incluído) — satisfaz a exigência OSI do vencedor.
6. **Desempate por antiguidade:** submeter candidatos fortes cedo, não só no último dia.

## Esclarecimentos do canal #duvidas (transcritos 18/09)

- **Dados externos — escopo confirmado (12h40):** outras variáveis ERA5, outras reanálises e dados
  de observação são permitidos, "desde que bem documentados"; a comissão avalia as submissões ALÉM
  da métrica final.
- **Autorregressão (12h46):** "a previsão será sempre para o mês seguinte"; modelo autorregressivo
  ou não é decisão da equipe → usar previsões próprias como input para alvos de 2024 é legal.
- **Copernicus/SEAS5 com cadastro gratuito (19h02):** confirmado explicitamente — "são dados
  públicos, pode fazer o download e utilizar". Fecha a dúvida de acessibilidade do manifesto.
- **Escala temporal externa (1h19):** dados semanais/diários anteriores ao mês previsto são legais;
  áreas fora do domínio da América do Sul podem ser input.
- **Vazamento do teste_features (18h55–19h02):** reportado publicamente que os campos atmosféricos
  do mês-alvo de uma linha aparecem como time_origem da linha seguinte (23/24 alvos exploráveis).
  Resposta oficial: "nós sabemos dessa limitação, por isso vamos revisar as submissões feitas" —
  haverá revisão das submissões por leakage.
- **Entregáveis (9h38):** acesso ao código ao término; plataforma livre; **não haverá pitch**.
- **Prazo:** competição aberta até 23/09.
- **Certificado de participação** para todos os participantes.

## Esclarecimento adicional (19/09, canal #duvidas)

Alexandre C. A. levantou que os dados de avaliação (2023-24) são públicos via ERA5/Copernicus —
hiperparâmetros podem ser ajustados sobre o próprio teste. Resposta oficial (geronimo.lemos):

> "Sim, os dados são públicos... A forma como encontramos para tentar mitigar esse problema é
> revisar as submissões e criar um conjunto público e privado (embora ambos sejam públicos).
> [...] deixamos as regras bem claras [...] e esperamos que todos as sigam. Obviamente, faremos
> o que está ao nosso alcance para tentar garantir isso. Último comentário sobre overfit:
> a preocupação é genuína e isso é um problema em várias competições públicas do Kaggle."

**Consequência:** a organização reconhece oficialmente que a separação público/privado é nominal
(os dois anos existem no ERA5) — a única defesa real é a revisão de código/causalidade. Isso
eleva o valor de: (a) documentação de causalidade (test_invariance, DATA_MANIFESTO), (b) método
OOF-first com submits só de confirmação — auditável e defensável, (c) NÃO usar ERA5 2023-24 de
tp em qualquer etapa (inclusive seleção de submits = peeking no privado).
