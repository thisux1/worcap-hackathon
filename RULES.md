# Regras da Competição — WORCAP 2026 (transcrição da aba Rules + esclarecimentos oficiais)

Fonte: aba "Rules" da competição (acesso restrito a inscritos) + esclarecimentos do
organizador no canal oficial de dúvidas, consolidados em 18–19/09/2026.

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

- **Anúncio de abertura:** dado o estado atmosférico observado de um mês, prever a
  precipitação média do mês seguinte, em mm/dia, em cada ponto da grade sobre a América do Sul.
  Dez variáveis do ERA5 em 0,25°, de 1940 a 2022, para treino. A avaliação cobre 24 meses: 2023
  alimenta o placar público durante a semana e 2024 decide o resultado final. Métrica única:
  RMSE sobre a chuva absoluta. Equipes de até 4 pessoas. Dados externos liberados, desde que
  públicos e declarados.
- **Limite de envios:** anunciado inicialmente como 3/dia; corrigido oficialmente para 5/dia.
- **Regra de causalidade — esclarecimento explícito:** para prever um mês T, vale qualquer dado
  que em tese estaria disponível até o fim do mês T−1; nada posterior. Na prática, para prever
  fevereiro de 2023 pode-se usar tudo até janeiro daquele ano: os campos que já vêm no
  teste_features.nc, dados externos em escala diária ou semanal, índices climáticos, informação
  de fora do domínio. Consequência explicitada pela organização: os campos de janeiro de 2023,
  que vêm no arquivo como preditores de fevereiro, não podem ser usados para estimar a chuva de
  janeiro de 2023 — reorganizar o arquivo para diagnosticar o próprio mês não é previsão.

## Consequências operacionais para o pipeline

1. **Corte causal confirmado:** features para o alvo T limitadas a dados até fim de T−1. O
   pipeline já indexa por `time_origem = T−1` (ver `worcap/features.py`) — compliant por construção.
   Dados externos do próprio mês-alvo T são **proibidos** (resolve P-007: o default D-008 era o
   correto).
2. **Dados externos fora do domínio** explicitamente liberados — legitima ERSSTv5 (Pacífico/Atlântico
   globais), ERA5 MSLP Southern Ocean (fora da grade oficial), NMME/SEAS5 via C3S.
3. **Features diárias/semanais externas** são legais até fim de T−1 (não explorado — mensal basta).
4. **NMME lead-1:** forecast com init no mês T−1 prevendo T está disponível ~dia 8–10 de T−1
   (real-time), dentro da janela — legal pela regra "em tese disponível até fim de T−1".
5. **Licença:** repo licenciado MIT (LICENSE incluído) — satisfaz a exigência OSI do vencedor.
6. **Desempate por antiguidade:** submeter candidatos fortes cedo, não só no último dia.

## Esclarecimentos adicionais do canal oficial (18/09)

- **Dados externos — escopo confirmado:** outras variáveis ERA5, outras reanálises e dados de
  observação são permitidos desde que bem documentados; a comissão avalia as submissões além da
  métrica final.
- **Autorregressão:** a previsão é sempre para o mês seguinte; usar ou não um modelo
  autorregressivo é decisão da equipe — previsões próprias como input para alvos de 2024 são
  legais.
- **Copernicus/SEAS5 com cadastro gratuito:** confirmado explicitamente que são dados públicos,
  livres para download e uso. Fecha a dúvida de acessibilidade do manifesto.
- **Escala temporal externa:** dados semanais/diários anteriores ao mês previsto são legais;
  áreas fora do domínio da América do Sul podem ser input.
- **Vazamento do teste_features:** foi reportado publicamente que os campos atmosféricos do
  mês-alvo de uma linha aparecem como time_origem da linha seguinte (23/24 alvos exploráveis).
  A organização respondeu que já conhecia a limitação e que revisaria as submissões por leakage.
- **Entregáveis:** acesso ao código ao término; plataforma livre; sem pitch.
- **Prazo:** competição aberta até 23/09.
- **Certificado de participação** para todos os participantes.

## Esclarecimento adicional (19/09, canal oficial)

Foi levantado no canal oficial que os dados de avaliação (2023-24) são públicos via
ERA5/Copernicus — hiperparâmetros poderiam ser ajustados sobre o próprio teste. A resposta da
organização: sim, os dados são públicos; a mitigação adotada é a revisão das submissões e a
separação entre conjunto público e privado (embora ambos sejam públicos), com a expectativa de
que as regras sejam seguidas e a ressalva explícita de que overfit é uma preocupação genuína e
comum em competições públicas do Kaggle.

**Consequência:** a organização reconhece oficialmente que a separação público/privado é nominal
(os dois anos existem no ERA5) — a única defesa real é a revisão de código/causalidade. Isso
eleva o valor de: (a) documentação de causalidade (test_invariance, DATA_MANIFESTO), (b) método
OOF-first com submits só de confirmação — auditável e defensável, (c) não usar ERA5 2023-24 de
tp em qualquer etapa (inclusive seleção de submits = peeking no privado).
