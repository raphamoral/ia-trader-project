# Histórico do projeto: AI Trader (B3 + Binance)

A arquitetura segue o método AOEN (`R:\WORK\ofhunter\aoen-skill`), que deriva a arquitetura das decisões de negócio.
No AOEN, "cliente" significa **estratégia/ativo**, porque hoje o sistema é para uso próprio.

## Forma do sistema
Entram preços e notícias com o horário em que foram publicadas. Esses dados viram uma visão do mercado naquele instante, sem nenhuma informação do futuro. Essa visão vira uma decisão (comprar, vender ou ficar de fora, com tamanho), que passa por um filtro de risco antes de virar ordem na corretora.

## 1. Núcleo e mercado
- **Núcleo:** o motor de decisão, que recebe a visão do mercado e devolve um sinal (o "agente isolado").
- **Por que é este:** coleta, corretoras e painel são periferia. Sem o motor, o sistema não tem razão de existir.
- **Isolamento:** contrato `Engine.decide(MarketView) -> Signal` (`trader/core/engine.py`). Dá para trocar LightGBM, regras evoluídas, transformer ou LLM sem mexer no resto.
- **Mercado:** uso próprio, com capital próprio. A dor é operar com disciplina e sem emoção.

## 2. Interfaces
- **Linha de comando e agendador** (`python -m trader ...`): coleta, treino, backtest e evolução. É a interface atual.
- **Painel web (futuro):** posições, PnL, árvore genealógica dos agentes. É onde um RAG ajuda, explicando decisões.
- **Corretoras:** adaptadores atrás do contrato `Broker` (Binance agora, depois MT5/Nelogica). O núcleo não sabe em qual corretora opera.
- **Camada de serviços:** risco e orquestração não sabem se quem chamou foi um backtest, uma simulação ou o modo real.

## 3. Configuração
- **Fluxo fixo:** coletar → gerar os indicadores → decidir → aplicar o risco → executar → registrar.
- **Parâmetros por estratégia/agente:** o genoma (estratégia, parâmetros, tamanho, filtros, cooldown), o ativo, o timeframe e os limites de risco.
- **Agente ou ativo novo:** é uma linha nova de configuração ou do registro, sem deploy.

## 4. Custo e isolamento
- **Isolado:** cada agente roda e é medido separadamente. A LLM de notícias e do Criador roda em processo próprio, uma vez por evento, e o resultado é compartilhado entre os agentes.
- **Unidade de custo:** a operação (trade). Os modelos locais custam praticamente só energia.
- **Conta (Binance spot, 0,1% por lado):** uma operação de R$1.000 paga 0,2% entre entrada e saída, ou seja, **R$2,00**. Com 10 operações por dia: R$20 por dia × 30 dias = **cerca de R$600 por mês só de taxa**.
- **Conclusão:** o modelo precisa ganhar, em média, **mais de 0,2% por operação** só para empatar.
- **B3 (WIN e WDO):** os custos por contrato ainda estão em aberto.

## Monetização
Hoje o objetivo é lucro próprio. Se um dia virar produto (venda de sinais ou copy-trade), as interfaces e a configuração por estratégia já estão preparadas, e o custo por estratégia já é medido.

## Pilha tecnológica
- **Linguagem:** Python 3.12, rodando direto no Windows (`.venv`, dependências no `Pipfile`).
- **Dados:** `ccxt` para a Binance, Parquet para guardar os candles (um arquivo por ano), `pandas`/`numpy`, `pytest`.
- **LLM:** local via Ollama (Qwen2.5-3B cabe na RTX 3050 de 6 GB), acessada por uma API no formato da OpenAI.
- **Por quê:** as escolhas vêm do hardware (RTX 3050 Laptop 6 GB, 20 núcleos), de manter o custo de APIs em zero e da necessidade de trocar o motor sem reescrever o resto.

---

## Decisões (2026-09-28)

1. **Nenhuma LLM treinada do zero; núcleo híbrido.** Treinar do zero é caro e fica pior que os modelos abertos. LLMs também não servem para séries de preço: as respostas variam e a execução é lenta. O motor é quantitativo e evolutivo. A LLM local só entra para interpretar texto (notícias e autópsias).
2. **MVP começa pela Binance.** Os dados são grátis, existe testnet para simulação e o mercado funciona 24/7.
3. **Dados da B3:** agora, baixados direto da B3 (arquivos em `data\raw\b3\`). A Nelogica (ProfitDLL) entra no futuro para dados intraday tick a tick.
4. **Agentes evolutivos.** Cada agente se retroalimenta. Se falhar, morre: uma autópsia registra a causa (drawdown, taxas, sem vantagem, estagnação) e nasce um sucessor com uma mutação dirigida por essa causa. Quem desenha o sucessor é o **Criador**, um agente que não morre: primeiro funciona por regras, depois por LLM. Quem acerta **ganha uma vaga e passa a ter 2** agentes (ele e um filho).
   - O número total de vagas tem teto, definido pelo processamento disponível. Com a população cheia, o pior agente é removido.
   - Nenhuma linhagem ocupa mais de cerca de 30% das vagas, para manter a diversidade.
   - **Contra sobreviventes por sorte:** mínimo de operações antes de julgar, resultado medido fora da amostra e depois das taxas, precisa superar o motor de referência, e um período final de dados (holdout) nunca é usado na seleção.
5. **Risco fora da evolução.** O `RiskGate` tem limites congelados (tamanho máximo da posição, stop diário, parada de emergência). Nenhum agente consegue alterá-lo. A seleção natural acontece só na simulação; no dinheiro real opera um campeão, com aprovação manual.
6. **LLM local no Ollama.** Ajuste fino (QLoRA) só mais tarde e na nuvem (AWS g5 ou RunPod, algumas horas), quando o cemitério tiver autópsias suficientes para servir de dados de treino.
7. **Roda direto no Windows.** A biblioteca do MetaTrader 5 e a ProfitDLL só funcionam no Windows, o WSL estava limitado a 7 GB de RAM e o disco `R:` é lento pelo WSL. O código é portável para rodar o treino na nuvem Linux.
8. **turbovec** (índice vetorial com TurboQuant) fica reservado para a memória de notícias e autópsias, atrás de um contrato `Memoria.buscar`. Não é necessário agora: o volume de dados é pequeno e ele não acelera a evolução.
9. **Acesso à LLM pela API no formato da OpenAI.** Assim o sistema roda com **zero tokens do Claude**. O `LLMCreator` chama a LLM local ou remota direto e volta às regras se ela falhar. O usuário usa o openclaude como sua interface para conversar com a LLM.

## Decisões de 2026-09-29

10. **Regras permanentes.** O Claude nunca faz commit nem aparece como autor (hook `commit-msg` bloqueia). Treino e execução de LLM com dados do projeto são sempre na LLM local (Ollama) ou na LLM remota do usuário, nunca no Claude.
11. **Treino da LLM adiado por falta de sinal.** Numa verificação de sinal em 290 autópsias, as correções por regras empataram com mutação aleatória (~50%) e o pai mantido vivo rendeu o mesmo que o sucessor. Só com preço, não há o que ensinar. O treino volta quando a verificação de sinal (`python -m trader.training signal-check`) mostrar vantagem.
12. **Criador com LLM melhor que regras no swing 4h:** nota média 2,88 × 1,47 nos 180 dias finais (3 sementes, 3 de 3 a favor). É indício, não prova (12,5% de chance por acaso).
13. **Bug do rebalanceamento corrigido.** O RiskGate reajustava a posição a cada candle e pagava taxa sempre; agora há uma faixa de tolerância (`rebalance_band`).
14. **Seleção por retorno ajustado ao risco** (retorno ÷ maior queda) e exigência de 2 janelas seguidas acima da referência para prosperar.
15. **B3:** COTAHIST de 2023 a 2026 importado (10 papéis), com ajuste de desdobramento (BBAS3 1:2 em 2024). COTAHIST não tem WIN/WDO; para os minicontratos é preciso Nelogica ou MT5.
16. **Notícias:** pipeline white em várias línguas com inglês como pivô, reputação por fonte sem olhar o futuro, texto completo das notícias importantes (respeitando robots.txt, sem contornar paywall). Histórico pelo GDELT em arquivos brutos (~7 TB de tráfego, ~24 dias); a API do GDELT bloqueou o IP e não se contorna bloqueio.
17. **Black:** fontes duvidosas legais e separadas (prefixo `black_`), monitor de hacks, dados alternativos, análise de antecipação × eco. Nada de login, dark web fechada ou coleta proibida pelos termos.
18. **Experimento ao vivo:** três forks do mesmo ponto de partida — contínuo (só preço), sem black (white) e tudo (white + black) — operando em papel 24h, controlados pelo painel local (http://127.0.0.1:8765).
19. **Por estilo, o tempo gráfico certo:** swing em 1h/4h, day trade em 5m (e 1m), scalper em 1m com ordens limitadas. Candle de 15m não serve para day trade.
20. **Taxa é o gargalo no intraday:** a 0,1% por lado (spot), a evolução em 15m aprendeu a operar pouco (~700 mortes por taxa). Testar futuros com ordem limitada (0,02%) e a B3 (custo por contrato baixo).
21. **Memória do Criador:** galeria dos que prosperaram (com análise de sucesso e contexto de mercado) + cemitério, ao vivo e do histórico (mesmo ativo, tempo gráfico e taxa). RAG dos livros (híbrido, com metadados e busca agêntica com limite) entra quando o índice existir.
22. **Objetivo pessoal: estabilidade.** Uso próprio; priorizar consistência e preservação de capital (escada de capital, sem ruína).

## Decisões de 2026-09-30 a 2026-10-01

23. **Recursos do PC são limite rígido.** O sistema inteiro usa no máximo 3 GB de RAM, cada job no máximo 1,2 GB e deve sobrar pelo menos 1,5 GB livres. Os processos rodam em prioridade abaixo do normal. O guarda de memória do painel pausa os jobs na ordem scalp → gdelt → sem black → tudo → contínuo; a tendência e a coleta nunca são pausadas. Tudo é instalado no `R:` (o `C:` está cheio): os modelos do Ollama ficam em `R:\Ollama\models` e o Tesseract em `R:\Tesseract-OCR`.
24. **Causa dos travamentos: leitores de notícias carregavam o histórico inteiro.** O texto completo chegava a 2,2 GB e o `load_trusted` do black passava de 5 GB por job. A correção:
    - leituras limitadas ao período;
    - janela confiável de 35 dias;
    - cache de features black (`data/features/black/`), calculado na coleta a cada 15 minutos.
25. **Robustez no Windows.**
    - `replace_with_retry` e `unique_tmp` resolvem a troca de arquivo bloqueada por leitores e a colisão de `.tmp`, que já tinha corrompido o SOL 5m de 2026.
    - SQLite compartilhado entre threads usa uma trava.
    - O runner sai depois de 10 erros seguidos para o `.bat` religá-lo com o código novo. Um import antigo tinha congelado o scalper por um dia.
26. **Achados de pesquisa (com holdout):**
    - Análise técnica só com preço não tem vantagem em swing nem em day trade, nem com taxa de 0,02%.
    - Genes de notícia: empate (6 sementes, −0,03 × −0,05) e mais risco.
    - Momentum entre moedas e funding: pioraram o resultado.
    - Estratégias criadas pela LLM (61 propostas numa DSL segura) não passaram no holdout de 2024–2026.
    - Alarme de hack ≥ US$100M com 24h de pausa reduz pouco a maior queda (24% → 21–23%) e custa 1 a 2 pontos de retorno. Fica como opção, desligado.
27. **Vencedora robusta: a tendência diária "comitê".** Ela combina as médias de 50, 100, 150 e 200 dias com volatilidade-alvo de 30%, em 5 moedas.
    - Histórico: 17%/ano, maior queda de 24%, pior mês −7,8%.
    - Holdout 2024–2026: 18,3%/ano, maior queda de 19,5%, Calmar 0,94.
    - Só o BTC: 49%/ano, maior queda de 47%, t = 2,54.
    - Roda em papel com R$1.000 (`trend_runner`, `data/live/tendencia.sqlite`).
28. **Livros (RAG).** O usuário indexou os livros por conta própria; os PDFs escaneados passam por OCR. O destilador gerou `playbook.json` com 32 regras. Só valem os números que aparecem nos trechos (ex.: RSI 14 em 30/70, filtro de tendência de 112 a 275). Esses números viram priors dos genomas.
29. **B3 em espera.** Já existe a base para WIN e WDO: vencimento, rolagem, série contínua, custos e proteção contra saldo negativo em gap. Os dados intraday dependem da Nelogica (ProfitDLL, preço a consultar). O foco agora é a Binance.
30. **Chat direto com a LLM** em http://127.0.0.1:8766 (`trader.chat`, atalho "AI Trader - Chat"). Ele independe do painel e do laboratório. Responde com o contexto do projeto, o resultado da tendência ao vivo, o playbook e trechos dos livros.
    - A LLM é útil para conversar, mas inventa fatos e capacidades. Exemplos: disse volatilidade-alvo de 20% (o certo é 30%), afirmou que "aprendeu com as perdas" e que roda testes, e sugeriu parâmetros sem teste.
    - Por isso nada que ela proponha é usado sem passar por backtest e holdout.
31. **Notícias precisam ser em tempo real** (websocket); a cada 2 minutos ou de hora em hora é tarde demais.
32. **Arquitetura alvo: reflexo (ms) + raciocínio (s).**
    - **Raciocínio:** a LLM local, com os livros, lê o contexto (notícias white e black, livro de ofertas, indicadores) e escreve uma **política** em JSON validado. A política diz quais reações estão aprovadas, os limites de risco e o viés.
    - **Reflexo:** executores em milissegundos aplicam só o que a política aprovou. Um classificador rápido de notícias, destilado de rótulos da Qwen, dispara reações pré-aprovadas sem esperar a LLM.
    - **Agentes:** executam apenas dentro da política. O ranking é por **lucro líquido**, com mínimo de operações; ruína mata o agente.
    - **Entradas em tempo real** (websockets da Binance):
      - preço;
      - livro de ofertas (depth@100ms: desequilíbrio, paredes, spread);
      - fluxo (aggTrades: CVD, ordens grandes);
      - indicadores incrementais: médias, MACD, ADX, Ichimoku, RSI, estocástico, CCI, Bollinger, ATR, Keltner, Donchian, VWAP, OBV, pivôs e padrões de candle;
      - **Fibonacci** a partir de topos e fundos confirmados (causal);
      - notícias white e black.
      - sugestões da Qwen (01/10), a testar: correlação entre os ativos; volatilidade implícita de cripto (DVOL da Deribit, o "VIX do cripto"); funding em tempo real; trailing stop e alvo de lucro ajustados pela volatilidade como regras da política.
    - **Retenção de dados** (não lotar disco nem RAM, local ou nuvem):
      - notícias brutas: 90 dias;
      - candles de 1m: 6 meses;
      - candles de 5m: 2 anos;
      - agregados: para sempre;
      - ticks: só em memória.
    - **Ordem de construção, uma peça por vez:**
      1. política + executores;
      2. websockets + indicadores + reflexo;
      3. fluxo histórico (data.binance.vision);
      4. classificador rápido;
      5. ranking por lucro + retenção.
    - Nada roda 24h antes de ser testado.
33. **Recusas mantidas:**
    - instalar ou operar o openclaude (o usuário usa por conta própria);
    - contornar o bloqueio de IP do GDELT;
    - dark web ou fontes com login;
    - indexar o acervo de livros (o usuário fez);
    - destilar respostas do Claude na Qwen.
34. **Saídas novas no day trade** (ideia do usuário). O stop nunca recua. Os valores ficam para a evolução decidir e precisam passar no holdout.
    - **Trava de lucro em degraus** (`lock_at`, `lock_keep`): depois de um ganho de X R, o stop sobe para garantir parte do ganho, nunca abaixo do zero a zero com as taxas. Exemplo: entrou a 100, alvo 150, chegou a 125, o stop vai para 110.
    - **"Deixa correr"** (`runner_vol`, `runner_atr`): com a posição no lucro e a volatilidade alta, o alvo é cancelado e o stop só sobe. A distância é de N ATRs, nunca menor que o stop inicial.
    - O chat ganhou regras de honestidade: sem memória, não roda testes, não inventa capacidades e marca toda sugestão como "hipótese a testar".
35. **Random Forest no diário: reprovado** (ideia de um artigo da Coinmonks; `python -m trader.research.rf_daily`, 2 min).
    - Configuração: indicadores do artigo (retornos, médias, RSI, MACD, Bollinger, volume, volatilidade), 5 moedas, walk-forward com retreino a cada 90 dias, parâmetros fixos, 3 sementes, mesmas regras de execução e tamanho do comitê.
    - Acerto do dia seguinte: 53% na seleção (AUC 0,55) e 50,2% no holdout (AUC 0,51), ou seja, moeda ao ar.
    - Holdout 2024–2026: de −1% a −10% ao ano, contra +18,3% do comitê. Mesmo na seleção, rende 4–8% ao ano contra 32% do comitê.
    - Conclusão: indicadores clássicos combinados por machine learning não preveem o dia seguinte. O que funciona é seguir a tendência de prazo longo.
36. **Saídas novas testadas: não ajudam** (`python -m trader.research.exit_test`, 40 agentes veteranos × 3 moedas × 2 períodos de 6 meses, teste pareado).
    - Os agentes já perdiam: mediana de −6%.
    - Trava: neutra, −0,2 ponto (t −0,6).
    - "Deixa correr": pior, −0,9 ponto (t −2,4).
    - Saída melhor não cria vantagem; os genes ficam disponíveis e desligados por padrão.
37. **Filtro do comitê (meta-labeling): promissor** (`python -m trader.research.meta_label`).
    - Um Random Forest olha só os dias em que o comitê está comprado e reduz a posição quando a chance de render nos próximos 10 dias é baixa.
    - Holdout 2024–2026: 15–17% ao ano contra 18,3% do comitê, mas maior queda de 10–13% contra 19,5%, Calmar 1,15–1,67 contra 0,94 e pior mês −3,5% contra −7,8%.
    - Controle: com a mesma exposição média, cortar posição ao acaso ou de forma constante dá Calmar de 0,6–0,9. O filtro ficou acima do percentil 90 do acaso em 4 de 6 casos.
    - AUC só 0,51: o sinal é fraco e o ganho vem de evitar alguns dias ruins.
    - Candidato para rodar em papel ao lado do comitê antes de substituí-lo.
38. **Peça 1 pronta: política + executor** (`trader/policy/`).
    - A Qwen lê o contexto (comitê, mercado de 5m, manchetes das últimas 6h, regras dos livros) e escreve a política em JSON (`python -m trader.policy write`, 60–100 s).
    - A validação só aceita limites mais apertados que os congelados, rejeita reações amplas ou repetidas e devolve os erros para a LLM corrigir. Se falhar, mantém a política atual ou cai na SAFE (sem entradas).
    - O executor aplica a política em microssegundos: viés por ativo, setups permitidos, tamanho pelo risco, stop, alvo, trava, "deixa correr", reações a notícias (zerar, reduzir, bloquear, apertar), limite de perda do dia e kill. Política vencida não abre posição nova.
    - Teste real: na 1ª versão a Qwen errou o formato e depois corrigiu, repetiu reações e a justificativa contradizia o viés; com o exemplo no pedido, saiu coerente na 1ª tentativa. Atenção: ela tende a copiar os valores do exemplo.
39. **Peça 2 pronta: tempo real** (`trader/realtime/`, `python -m trader.realtime watch --seconds 60 [--paper]`).
    - Websockets dos futuros da Binance. **A Binance separou os streams:** negócios em `/market/stream`, livro em `/stream`.
    - Livro (top 20 a cada 100 ms): spread, desequilíbrio, paredes.
    - Fluxo (aggTrade): CVD, agressão na janela, negócios grandes.
    - Candles de 1m montados dos negócios, com indicadores incrementais em O(1): médias, EMA, MACD, RSI, ATR e percentil, Bollinger, estocástico, Donchian, OBV, VWAP e Fibonacci de topos e fundos confirmados. Os testes conferem que batem com o pandas e não olham o futuro.
    - Cada negócio passa pelo executor (stops no tick).
    - Reflexo de notícias por palavras-chave e valor em US$ (provisório, até o classificador da peça 4).
    - Ticks só em memória; o agregado do minuto fica no máximo 1 dia na RAM.
    - Teste ao vivo: 6.253 mensagens em 150 s. **O relógio do PC está ~7 s adiantado** em relação à Binance: sincronizar antes de operar.
40. **Peça 3: livro de ofertas e fluxo NÃO dão vantagem ao day trade depois das taxas** (2026-10-02).
    - Dados: futuros de SOL, XRP e ETH, mar–set/2026, baixados de data.binance.vision (5,7 GB) e guardados só como resumo por minuto (70 MB em `data/flow/`) por `python -m trader.flowhist.download`.
    - Teste: `python -m trader.research.flow_edge`, decisão a cada 5 min, 1 min de atraso na entrada.
    - Parte A: nos quintis extremos de agressão, ordens grandes e desequilíbrio do livro, a diferença de retorno nos 15 min seguintes é de **~0,5 a 1,2 ponto-base**. O custo de ida e volta é de 4 (maker) a 10 (taker) pontos-base. O sinal mais forte é o volume anormal (+1 a +2,7 bps), ainda abaixo do custo.
    - Parte B: modelo walk-forward mensal, horizontes de 5, 15 e 60 min, custos maker e taker. **Perdeu nas 3 moedas e em quase todos os meses** (t de −2,4 a −7,5).
    - Ressalva: o livro histórico é grosso (profundidade por faixa de %, a cada ~25 s). A microestrutura em milissegundos só pagaria com latência de servidor ao lado da corretora e ordens maker garantidas, e nós temos internet doméstica e Python.
    - Conclusão: day trade direcional com essa informação não é caminho de lucro para nós.
41. **Peça 4: classificador rápido construído, mas NÃO aprovado** (`trader/realtime/news_labels.py`, `news_model.py`).
    - A Qwen rotulou 1.076 manchetes de cripto de 2023–2026 (RSS + manchetes tiradas do endereço das notícias do GDELT) no formato do reflexo.
    - Os rótulos saíram inconsistentes: usou "any" como tipo 147 vezes e "outage" 109 vezes.
    - O classificador destilado (TF-IDF + regressão logística, ~3 ms por manchete) ficou perto do chute no teste de 2026: relevante 75% contra 78% da classe mais comum, tipo 29% contra 20%, direção 40% contra 34%, 0 de 5 notícias graves pegas.
    - Fica salvo como candidato (`data/models/news_reflex_candidate.pkl`). O reflexo segue nas palavras-chave, e o modelo só entra em uso se superar a maioria e pegar notícias graves.
    - Para melhorar: rótulos melhores, com tipo por regra e a Qwen só para sentimento e gravidade, mais exemplos ou um modelo local maior.
42. **Comitê × comitê com filtro, lado a lado em papel** (2026-10-02).
    - Job novo no painel, `tendencia_filtro` (`trader.live.trend_filter_runner`, banco `data/live/tendencia_filtro.sqlite`): o mesmo comitê, com a posição escalada pelo filtro do item 37 (variante gradual 0,4–0,6). O Random Forest é retreinado a cada dia só com o passado (~5 s, ~250 MB).
    - Começou em 01/10 com R$1.000; a tendência pura começou em 30/09. Para comparar, use os retornos a partir de 01/10.
    - Os dois jobs são leves e a guarda de memória nunca os pausa.
    - **Bug corrigido no painel:** o Windows reaproveitou o PID antigo da tendência para o NVIDIA Overlay. O painel achava que a tendência estava ligada e, ao desligar, derrubaria o Overlay. Agora o PID só conta, e só é derrubado, se a linha de comando for a do job (teste em `tests/test_panel_guard.py`).
43. **Universo do comitê: mais moedas piora; ouro melhora** (2026-10-07, `python -m trader.research.universe_test`).
    - 47 moedas com diário da Binance desde 2018, **incluindo deslistadas** (LUNA, FTT, WAVES…), universo escolhido todo mês só com o passado.
    - Holdout 2024–2026, retorno anual / maior queda / Calmar: 5 atuais 18% / 19,5% / 0,94; top10 11% / 17% / 0,69; top20 9% / 18% / 0,50; top30 7% / 20% / 0,34; 10 de menor correlação ao BTC 5% / 21% / 0,22.
    - Mais moedas = menos retorno e a mesma queda: as moedas pequenas têm tendência pior e caem juntas na crise. Escolher por baixa correlação foi o pior: a correlação do passado não prevê a da próxima queda. Parte da vantagem das 5 atuais é viés de sobrevivência (foram escolhidas sabendo quem deu certo).
    - **5 atuais + ouro (PAXG):** 19,9% / **13,9%** / **1,43**, pior mês −5,9% (contra −7,8%). Na seleção 2019–2023 também reduziu a queda (23,7% → 20,3%) perdendo pouco retorno. Correlação do PAXG com o BTC: 0,14, a mais baixa de todas.
    - Ressalva: o ouro teve uma alta forte em 2024–2026; parte do ganho é desse período. A favor, seguir tendência em ouro tem décadas de histórico fora do cripto.
    - Diversificação de verdade vem de outra classe de ativo, não de mais criptos.
44. **Terceiro runner em papel: 5 criptos + ouro** (2026-10-07). Job `tendencia_ouro` no painel (`trader.live.trend_gold_runner`, banco `data/live/tendencia_ouro.sqlite`), R$1.000 desde 07/10, mesmo comitê e mesma vol-alvo, com PAXG/USDT na cesta. Os três comitês (puro, com filtro, com ouro) competem lado a lado; `TrendPaper` agora aceita a lista de símbolos.
45. **Comitê alavancado em futuros** (2026-10-07, `python -m trader.research.leverage_test`; taxa 0,05%, funding 0/8/15% ao ano sobre o notional, ruína se cair 90% num dia, só fechamentos diários).
    - Holdout 2024–2026, 5 criptos + ouro, funding 8%: 1x 17%/ano, queda 15%; **1,5x 26% / 22%**; **2x 34% / 29%**; 3x 50% / 41%; 5x 76% / 61%, pior mês −29%.
    - A nota (retorno ÷ queda) não melhora com alavancagem: ~1,2 em todos os níveis. Alavancar multiplica ganho e dor na mesma proporção, e o funding come 20–40% do retorno extra.
    - Melhor dia da carteira: 3,8% a 1x; só a 5x um dia chegou a 21%, com queda máxima de 61–80% no caminho. Não houve ruína no diário; intradiário seria pior, e PAXG tem perpétuo há pouco tempo.
    - Faixa razoável, se um dia for para o real: 1,5x a 2x. Acima disso a queda passa de 40% e o risco de liquidação intradiária cresce.
    - Pedido do usuário de "forçar a IA a achar 20% ao dia" foi recusado: já testado (`--fitness goal`, agentes morreram por ruína); forçar só gera overfitting.
46. **Autópsia do comitê: onde ele perde** (2026-10-09, `python -m trader.research.committee_autopsy`, 5 criptos + ouro, 2019–2026).
    - 345 operações: **72% são serrotes** (entra, o mercado vira em poucos dias, sai perdendo ~0,1% cada), 12% perdas lentas, 6% devoluções (ganhou e devolveu mais da metade antes de sair) e **10% ganhos grandes** que pagam tudo (+2,31 contra −0,49 de perdas, na escala da fatia de cada moeda).
    - É a anatomia clássica de seguir tendência: muitas perdas pequenas, poucos ganhos grandes. Eliminar TODAS as perdas por serrote (impossível) elevaria o bruto em ~25%: o teto do "entender os erros" é 17% → ~21% ao ano, não o dobro.
    - As quedas longas (2021–2023, 760 dias, −20% com BTC −74%; 2025–2026, −11% com BTC −48%) não são erro: é a estratégia protegendo. As quedas em mercado de lado (2019–2020, −18% com BTC +4%; 2024, −14% com BTC +38%) são o erro de verdade: serrotes em sequência.
    - Taxas custam só 1,1% ao ano. Exposição média de 30% do capital (vol-alvo). Captura assimétrica: +0,50%/dia quando o BTC sobe, −0,38%/dia quando cai: essa assimetria é a vantagem inteira.
47. **Correções para os serrotes: nenhuma é robusta** (2026-10-09, `python -m trader.research.committee_fixes`, 4 variantes fixadas antes de olhar o holdout).
    - Confirmação de 3 dias e histerese de 2% cortam os serrotes pela metade (72% → 38–50% das operações), mas no holdout rendem igual ou menos (13–19% contra 20% da base): o que elas filtram inclui o início das tendências grandes.
    - Saída rápida (média de 30 dias) e "fora quando lateral" (razão de eficiência): a primeira cria mais serrotes; a segunda foi péssima na seleção (Calmar 0,72) e ótima no holdout (2,2): inconsistente, não dá para confiar.
    - Conclusão: os serrotes são o preço de estar posicionado quando a tendência grande começa. Reduzi-los reduz os ganhos junto. O comitê base continua sendo a versão a usar; o filtro do item 37 (que aprende QUANDO o sinal costuma falhar, em vez de regra fixa) segue como a única melhoria candidata.
48. **Filtro v2 (ambiente de mercado): não é robusto** (2026-10-09, `python -m trader.research.meta_label_v2`).
    - Features novas, só com o passado: amplitude (fração das 47 moedas acima das médias de 200 e 50 dias), comitê do BTC e do ouro, funding do BTC (desde 2023; antes = 0) e volume anormal do mercado.
    - Holdout 2024–2026: v2 (RF) queda 7,5–7,9% e Calmar ~2,0, contra 10% / 1,6 da v1 e 13,9% / 1,43 do comitê puro. Parece ótimo.
    - Seleção 2019–2023: v2 Calmar 0,92 contra 1,27 da v1 e 1,39 do puro, e AUC 0,48 (abaixo do chute). O mesmo padrão da regra "lateral_fora": brilha num período e falha no outro. O funding só existe desde 2023, o que confunde a comparação.
    - Gradient boosting (HGB) foi pior em tudo.
    - Conclusão: o ambiente de mercado não dá um sinal estável de serrote. **A v1 continua sendo o único filtro consistente**: reduz a queda nos dois períodos (20% → 16% e 14% → 10%) ao custo de ~25% do retorno. Serrote segue sendo o preço de seguir tendência.
49. **Provedor de LLM como opção** (2026-10-09, a pedido do usuário, como teste). `trader/llm.py` centraliza as chamadas: Ollama local continua o padrão; `TRADER_LLM_PROVIDER=anthropic` (ou `--provider anthropic` no `python -m trader.policy write`) usa o SDK da Anthropic com `claude-fable-5-1` por padrão. A chave fica em `.env` (fora do git) ou `ANTHROPIC_API_KEY`. Cada chamada registra tokens e custo estimado (Fable: US$10/M entrada, US$50/M saída). Recusas de segurança viram erro e a API tenta um modelo alternativo (`fallbacks="default"`). O chat (8766) e o escritor de política aceitam o provedor; o rótulo da página mostra qual está ativo. Regra "só LLM local" passa a ser "local por padrão; Claude só quando o usuário pedir, com custo informado".
50. **Provedor `claude-cli` e o teste Qwen × Claude** (2026-10-09). O Claude Code em modo headless (`claude -p`, pelo WSL via `wsl.exe`, sem chave, com a assinatura do usuário) virou o terceiro provedor. `--tools ""`, `--no-session-persistence`, prompt de sistema próprio; o custo que o CLI estima fica no relatório. Mesma tarefa, mesmo contexto, a política de hoje:
    - **Qwen 2.5 7B:** 3 tentativas, 303 s, e nenhuma válida (inventou `sentiment: "neutral"` e `kind: "news"` e repetiu o erro depois de receber a lista de erros). Resultado: caiu na política anterior. Zero custo.
    - **Claude Fable 5.1 (CLI):** 1 tentativa, 40 s, válida. Leu o contexto de verdade: comitê 1,0 em BTC e SOL e 0,75 em ETH e XRP, XRP de fora (comitê mais fraco + maior vol + −7,6% em 7 dias), risco menor por ativo, setups restritos aos de tendência, limite diário apertado para 1,5%, 7 reações graduadas por tipo e gravidade de notícia. Estimativa do CLI: US$0,12 (coberto pela assinatura, mas conta no limite de uso).
    - Conclusão: para **escrever política** (raciocínio estruturado com schema), o modelo maior é claramente melhor e mais rápido; a 7B não sustenta o schema. Isso não muda o achado anterior sobre **lucro**: nenhum modelo cria vantagem onde os dados não têm; a política só define risco e reações. Decisão de uso fica com o usuário (limite da assinatura); padrão continua Ollama.
51. **Modelo local no nível do Claude para a política** (2026-10-09). Duas mudanças no escritor de política:
    - **Decodificação restrita:** no Ollama, o pedido leva o JSON Schema da política (`schema.json_schema`) e o modelo só consegue gerar tokens válidos: enumerações e estrutura ficam garantidas pela gramática. Faixas numéricas o Ollama não garante, então o parse `lenient` ajusta o número ao limite (nunca além dos congelados).
    - **Base por regras + só apertar** (`policy/rules.py`): o contexto vira uma política-base determinística (comitê → viés; vol e queda da semana → risco e "fora"; tabela fixa de reações). A LLM parte dela e o resultado final é o mais conservador entre a base e a resposta (`tighten`). Um modelo que erra só consegue deixar a política mais cautelosa; se a LLM falhar, vale a base.
    - Mesmo contexto de hoje: **Qwen 7B passou de 3 tentativas inválidas em 303 s para 1 válida em 44 s**, com a mesma estrutura da política do Claude (SOL e BTC comprados com risco pequeno, XRP fora, 7 reações graduadas). Diferença restante: a base deixa ETH fora (−8,6% na semana com comitê 0,75); o Claude manteve ETH comprado com risco menor. As duas são defensáveis; a base é mais conservadora por construção.
    - Lição: o que parecia "inteligência" do modelo grande era, em grande parte, disciplina de formato e regras que o contexto já determina. Com a estrutura vinda do código, o modelo local entrega o resultado esperado a custo zero; o Claude fica como opção para o julgamento fino.
52. **Carry do funding (cash-and-carry): reprovado para hoje** (2026-10-09, `python -m trader.research.carry_test`, funding de BTC/ETH/SOL/BNB/XRP desde 2020, margem 50%, custo 0,3% por liga/desliga).
    - O funding secou: média das 5 moedas por ano, anualizada: 2020 17%, 2021 33%, 2022 −9%, 2023 3,5%, 2024 10%, 2025 2,4%, 2026 1,3%.
    - Holdout 2024–2026: sempre posicionado rende 3,2%/ano (queda 0,4%); as variantes que ligam/desligam (média de 7 dias, hurdle, rotação) ficam **negativas** (−1,6 a −4%), porque o custo de trocar come um funding que já é pequeno.
    - Correlação com o comitê: 0,10 (diversifica), mas 50/50 reduz o retorno de 20% para 8% e a queda só de 14% para 10%. Não compensa.
    - Risco extra não modelado: dias com alta > 40% (liquidação da perna vendida): SOL 1, BNB 3, XRP 4 desde 2020.
    - Fica a regra: o carry só volta a interessar se o funding médio de 7 dias passar de ~8% ao ano por semanas seguidas (euforia). Script pronto para rechecar.
53. **Tendência com "holding" (sem alavancagem): segurar mais piora** (2026-10-10, 5 criptos + ouro, holdout 2024–2026, retorno/queda).
    - Comitê (sai gradual, referência): 20% / 14%. Entra 4/4 e só sai com 0/4: 16% / 19%. Entra 3/4 e sai só abaixo da média de 200d: 17,5% / 17%. Mesmo sem vol-alvo: 27% / 28%. Núcleo 50% sempre comprado + comitê no resto: 28% / 35%. Comprar e segurar: 31% / 54%.
    - Acumulação mensal (DCA) sem vender: 1,13–1,20× o aportado em 2024–2026 com queda de 48–58% do patrimônio/aportado; em 2019–2023, 5× com queda de 81%. O filtro de 200d na compra não reduz a queda (o problema é não vender).
    - Toda forma de segurar mais troca queda por retorno na mesma proporção ou pior; a nota (retorno ÷ queda) cai de 1,43 para 0,6–1,0. A saída gradual do comitê é o que protege.
54. **Rompimento de máxima/mínima semanal (ideia do usuário): mesma família, pior no holdout** (2026-10-10, 5 criptos, decisão diária: compra ao fechar acima da máxima das últimas N semanas completas, vende ao fechar abaixo da mínima; N = 1, 2, 4).
    - 2019–2023, com vol-alvo 30%: 1 sem. 31% / 25%; **2 sem. 38% / 29%**; 4 sem. 28% / 33%; comitê 32% / 24%. Parecido.
    - **Holdout 2024–2026**, com vol-alvo: 1 sem. 11% / 22%; 2 sem. 13% / 24%; 4 sem. 12% / 24%; **comitê 17% / 19,5%**. Em todas as versões o rompimento rendeu menos e caiu mais (Calmar ~0,5 contra 0,85).
    - Versão comprado/vendido (short no rompimento da mínima): ruim em todos os períodos (2–6% ao ano no holdout, 1–33% na seleção): o lado vendido em cripto só paga nos crashes e perde o resto do tempo.
    - Misturar comitê + rompimento (média): não melhora o comitê.
    - Leitura: rompimento semanal e comitê de médias são a mesma ideia (seguir tendência); o comitê ganha fora da amostra porque entra e sai em degraus (4 médias), enquanto o rompimento é tudo-ou-nada e leva o serrote inteiro.
55. **Rompimento semanal em degraus** (2026-10-10, 5 criptos, vol-alvo 30%). A) comitê de rompimentos de 1/2/4/8 semanas (25% cada): pior que o comitê de médias nos dois períodos. B1) +25% por semana que fecha acima da máxima anterior, −25% por semana abaixo da mínima: pior. **B2) +25% por semana acima da máxima, zera ao fechar abaixo da mínima ("entra devagar, sai de uma vez")**: queda menor nos dois períodos (24% → 19% e 19,5% → 11%) ao custo de retorno (32% → 24% e 17% → 11%); Calmar 1,25 / 1,00 contra 1,35 / 0,85 do comitê. É uma versão defensiva consistente, parecida com o filtro v1 em efeito, e não uma vantagem nova. Fica como opção de perfil; o comitê continua a base.
56. **"Segurar moedas emergentes" (ex.: SOL): viés de sobrevivência medido** (2026-10-10, universo de 47 moedas com deslistadas).
    - SOL desde set/2020: +3.298%, mas com queda máxima de **96%**; quem comprou no topo de nov/2021 está em −54% até hoje.
    - Cesta cega de todas as estreantes de 2020–21 do universo (16 moedas, compra 30 dias após a estreia, segurar até hoje): média +290% (puxada pela SOL), **mediana +77%, 6 de 16 abaixo da entrada**, todas com queda ≥ 92% no caminho (LUNA −84%, ICP −95%, FIL −96%).
    - Comprar tudo que existia em jan/2022 e segurar: **mediana −78%, só 5 de 42 positivas**.
    - O universo ainda é generoso (moedas conhecidas hoje); com todas as listagens da Binance a mediana seria pior. Ganho "estratosférico" é olhar a vencedora depois do sorteio.
57. **"Saber antes" qual listagem vai subir: não dá com dados públicos** (2026-10-10, `python -m trader.research.listings_study`; 681 listagens USDT da Binance desde 2017, inclusive deslistadas, diário em `data/research/universe_all/`; TVL da DeFiLlama em `data/research/tvl/`; 554 com resultado, 148 com TVL).
    - **Taxa-base por ano de listagem** (compra no dia 90, segura 12 meses, mediana): 2019 +22%, 2020 +128% (a loteria: 55% dobraram), **2021 −78% (85% perderam mais da metade)**, 2022 −39%, 2023 −9%, 2024 −81%, 2025 −66%. Fora de 2019–2020, a listagem típica perde a maior parte do valor no primeiro ano.
    - **Sinais no dia 90:** momentum, crescimento de volume e nível de volume preveem **ao contrário** (quanto mais quente, pior: Spearman negativo em 6–7 de 7 anos): é o padrão "sobe na listagem, despenca depois". Queda desde a máxima prevê a favor (6/7 anos) pelo mesmo motivo. **TVL (uso real) não prevê nada** (3/5 anos, medianas de −56% a −74% em todos os quintis). Regime do BTC na decisão: indiferente.
    - Mesmo o melhor quintil tem mediana negativa; a média positiva vem de poucos bilhetes premiados de 2020.
    - **Saída por tendência** (sem teto de ganho, sai na quebra da média de 50 dias): evita os −80%, mas a mediana fica em −1% a −8% e só 0–7% dobram.
    - **Cesta walk-forward** (sinais escolhidos em 2019–22, top-5/10 por ano em 2023–26): picks −3% de mediana contra −3% do resto. Zero vantagem.
    - Conclusão: a receita não existe em dados públicos. O que existe é o oposto: evitar listagens quentes. A fatia "emergentes" fica fora.
58. **Painel parou de responder: varredura de processos lenta** (2026-10-20). Cada `/api/status` varria todos os processos do Windows (16 s medidos naquele dia) e a página pede a cada 5 s: os pedidos se acumulavam e nenhum terminava, embora a página estática abrisse. Correção: uma única varredura com cache de 10 s compartilhada por status, guarda de memória e prioridade (`_procs`), status em cache por 3 s para vários navegadores, invalidação ao ligar/desligar job e timeout do Ollama de 1,5 s. Os três comitês em papel não foram afetados (seguiram gravando todo dia).

59. **Preparação para vender licença do código no DataVendor (datavendor.ai)** (2026-11-01). Marketplace da HUD: licença não exclusiva para laboratórios de IA, só repositório privado, taxa de 20% na retirada. Feito: `README.md`, `.env.example`, `Pipfile` com as 6 dependências que faltavam, e-mail e caminho pessoal movidos para o `.env`. Pendente do usuário, por ordem: decidir se vende, confirmar se o código escrito com o Claude pode ser licenciado para treino de IA, commitar o trabalho e rodar o estimador gratuito. Rascunho do anúncio em `data/datavendor/anuncio.md`.

## Estado atual (2026-10-01)

**Ligados (2026-10-07):** `tendencia`, `tendencia_filtro` e `tendencia_ouro` em papel, o painel (8765), o chat (8766) e o Ollama. Day trade, scalper, coleta e GDELT continuam desligados.

**Pronto:**
- núcleo, risco e backtest;
- evolução com Criador por LLM, galeria, lições e priors dos livros;
- B3 (COTAHIST, contratos WIN/WDO);
- notícias white e black com cache;
- RAG dos livros e playbook;
- day trade (estilos, regimes, escada);
- pesquisa: tendência diária, DSL e overlay de notícias;
- modo ao vivo em papel;
- painel com guarda de memória (porta 8765);
- chat (porta 8766);
- cerca de 230 testes.

**Pendente:**
- Sincronizar o relógio do Windows (`w32tm /resync`, como administrador).
- Peça 4: melhorar os rótulos e retreinar (o classificador atual não foi aprovado). Peça 5: ranking por lucro líquido + retenção.
- Decidir o foco: o comitê (com ou sem filtro) é a única vantagem comprovada; o day trade direcional não pagou as taxas em nenhum dado testado.
- Fonte de notícias em tempo real de verdade (hoje o RSS é lido a cada 2 min).
- Backfill do GDELT a partir de 2023: parado em ~26% (out/2025 a ago/2026 completos). Depois, testar o alarme de notícias em 2023–2026.
- Usuário:
  - desinstalar o Tesseract do `C:`;
  - exclusão ou remoção do McAfee;
  - preço da ProfitDLL (corporativo@nelogica.com.br).
  - DataVendor: ver item 59.

## Aprendizados (conceitos usados)
- **Walk-forward e holdout:** avaliar sempre em período nunca visto pela seleção.
- **Point-in-time / lookahead bias:** a notícia só existe para o agente depois de visível (publicação + latência, ou depois da leitura do texto completo).
- **Multiple testing:** com muitos agentes, alguém "acerta" por sorte; por isso várias sementes, mínimo de operações e verificação de sinal.
- **Algoritmo evolutivo com LLM como operador de mutação:** autópsia → sucessor; análise de sucesso → filho.
- **RAG:** simples, híbrido, com reranqueamento, com metadados, GraphRAG, agêntico.
- **Experimento controlado:** mesmo ponto de partida e mesmos dados; muda uma variável por grupo.
- **Métrica ajustada ao risco:** retorno ÷ maior queda.
- **Holdout de verdade:** estratégia que só brilha na seleção (DSL da LLM, genes de notícia) é ruído; a tendência passou porque sobreviveu ao período nunca visto.
- **Ranking por lucro líquido, não por acerto:** acertar muito com ganhos pequenos e perdas grandes perde dinheiro.
- **Núcleo substituível (AOEN):** contratos `Engine`, `Broker`, `RiskGate` e `news_features` permitem trocar o motor sem reescrever o resto.
