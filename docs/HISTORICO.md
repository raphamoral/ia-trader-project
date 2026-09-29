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

## Estado atual
**Módulos existentes:**
- `trader/domain.py`: tipos e `MarketView`, que só mostra candles já fechados.
- `trader/core/`: o contrato `Engine` e o motor de referência (cruzamento de médias).
- `trader/risk/gate.py`: `RiskGate` e `RiskLimits`.
- `trader/execution/`: o contrato `Broker` e o `PaperBroker` (taxa de 0,1%, sem venda a descoberto).
- `trader/storage/candles.py`: Parquet por ano, sem duplicar linhas.
- `trader/ingest/binance.py`: coleta paginada e incremental. O comando `python -m trader collect` já foi rodado com BTC/USDT em 1d e 4h desde 2020.
- `trader/backtest.py`: backtest barra a barra (decide no fechamento, executa na abertura seguinte) e janelas walk-forward.
- `trader/evolution/genome.py`: o genoma, as mutações e o `GenomeEngine`.
- `trader/evolution/fitness.py`: avaliação por janela, veredito (morrer, viver ou prosperar) e causas da morte.
- `trader/evolution/creator.py`: `RuleCreator` e `LLMCreator` (API no formato da OpenAI, com volta às regras).

**Pendente:**
- População (vagas e linhagem), o registro em SQLite, o comando `evolve` e os testes dos módulos de evolução e de backtest.
- O leitor dos arquivos da B3.
- A configuração do Ollama com o Qwen2.5-3B.

**Questões em aberto:**
- TurboQuant: qual é o uso exato. O link veio cortado; `turbovec` foi avaliado e reservado.
- Custos da B3 por contrato (WIN e WDO): emolumentos e corretagem.
