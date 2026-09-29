# Day trade alavancado: integração com o núcleo

Este módulo foi escrito sem alterar nenhum arquivo existente, porque o núcleo
estava sendo modificado em paralelo. Abaixo estão as mudanças propostas para
integrá-lo depois, e as premissas de mercado usadas.

## Premissas de mercado (Binance USDⓈ-M, conferidas em 2026-09-29)

| Item | Valor usado | Fonte |
|---|---|---|
| Taxa taker VIP 0 | 0,05% por lado (maker 0,02%) | [Binance: USDⓈ-M Futures Trading Fee Rate](https://www.binance.com/en/fee/futureFee); o ccxt `load_markets()` também retorna taker 0.0005 e maker 0.0002 |
| Funding | a cada 8h, às 00:00, 08:00 e 16:00 UTC; comprado paga e vendido recebe quando a taxa é positiva | [Binance FAQ: funding rates](https://binance.com/en/amp/support/faq/360033525031), [anúncio sobre funding](https://www.binance.com/en/support/announcement/important-updates-on-funding-rates-of-usd%E2%93%A2-m-perpetual-contracts-98d6b24d3e5c4f84a8ed04087997d8d0) |
| Histórico de funding | **real**, via `fetch_funding_rate_history` (público), salvo em `data/funding/binanceusdm/`. Sem arquivo, o backtest usa 0,01% constante a cada 8h | ccxt |
| Margem de manutenção | 0,4% (1ª faixa do BTCUSDT). **É uma premissa:** as faixas oficiais exigem chave de API (`fetch_leverage_tiers`). Confira na página *Leverage & Margin* da Binance | — |
| Liquidação | margem isolada inteira perdida, sem taxa adicional | simplificação |
| Ordem mínima | nocional mínimo **e** quantidade mínima (`MIN_NOTIONAL`, `MARKET_LOT_SIZE.minQty`) lidos em `GET /fapi/v1/exchangeInfo` em 2026-09-29: BTCUSDT 50 USDT / 0,001 BTC; ETHUSDT 20 / 0,001; SOL 5 / 0,01; XRP 5 / 0,1; DOGE 5 / 1. Veja `market.py` | [BTCUSDT: mínimo reduzido de 100 para 50 USDT em 2026-04-14](https://longbridge.com/en/news/282143186), [regra de nocional mínimo](https://www.binance.com/en/support/announcement/new-minimum-order-notional-limit-rule-for-usd%E2%93%A2-m-futures-76719bbaeeb847bbac4daa2906fcdcc0), [especificações dos contratos](https://www.binance.com/en/support/faq/usd%E2%93%A2-margined-futures-contract-specifications-360033161972) |

### R$100 no BTCUSDT não funciona

R$100 ÷ 5,5 = 18,18 USDT. No teto de 3x, isso dá 54,5 USDT de nocional. A
menor ordem de BTC, a ~83 mil USDT, é 0,001 BTC ≈ 83 USDT: o lote mínimo pesa
mais que a regra de 50 USDT. Operar exigiria 4,7x de alavancagem já na
primeira ordem, sem sobra nenhuma. O CLI checa isso na partida
(`feasibility`) e recusa rodar. SOL, XRP e DOGE têm mínimo de 5 USDT e
funcionam com R$100.

Algumas contas mudaram a frequência de funding de certos contratos para 4h ou
1h. Para o BTCUSDT, o histórico baixado mostra 8h.

## Mudanças propostas no código compartilhado

1. **`trader/evolution/population.py`: tornar a população genérica.** Hoje ela
   fixa `evaluate`, `random_genome`, `BASELINE` e `Cause`. A proposta é
   injetar um objeto "espécie":
   ```python
   class Species(Protocol):
       def random(self, rng) -> Genome: ...
       def evaluate(self, genome, candles, start, end) -> WindowStats: ...
       def extra_death(self, window_result) -> str | None: ...  # ex.: "liquidation"
       baseline: Genome
   ```
   Assim, `trader/daytrade/evolution.py::DayPopulation` (cerca de 150 linhas
   duplicadas) desaparece.

2. **`trader/evolution/fitness.py`:**
   - Adicionar `Cause.LIQUIDATION = "liquidation"`.
   - Tornar `_drawdown_context` público (`drawdown_context`), porque o day trade
     já o importa.

3. **`trader/core/engine.py`:** adicionar um caminho vetorizado opcional:
   ```python
   class VectorEngine(Protocol):
       def signals(self, candles: pd.DataFrame) -> np.ndarray: ...  # causal
   ```
   O backtest barra a barra com a `MarketView` é lento demais para 131 mil
   candles de 15m em cada geração. Os sinais vetorizados têm um teste de
   causalidade (`test_signals_are_causal`) que deveria valer para qualquer
   motor vetorizado.

4. **`trader/evolution/creator.py`:** o `LLMCreator` só conhece o genoma
   spot. Para o day trade, parametrizar o prompt (estratégias, genes,
   causas). Até lá, `DayRuleCreator` é o único Criador do day trade.

5. **`trader/__main__.py`:** expor `python -m trader daytrade ...` delegando
   para `trader.daytrade.__main__.main`, ou manter separado.

6. **B3 (WIN/WDO), depois:** o gate já aceita `session_end`. Faltam:
   - O horário do pregão, com zeragem compulsória perto do fechamento.
   - Custos por contrato (emolumentos e corretagem) em vez de percentual.
   - Margem de garantia da B3 em vez de alavancagem/mmr.
   - Ajuste diário em vez de funding.

   A proposta é implementar isso como outra `FuturesPaperBroker` atrás da mesma
   interface.

## Genes de autonomia e escalada de risco

Todos são genes, e o Criador (regras ou LLM) pode mudá-los. Os tetos
congelados do gate valem sempre.

| Gene | O que faz | Limite congelado |
|---|---|---|
| `risk` | fração do patrimônio perdida até o stop | `max_risk_per_trade` (2%) |
| `max_trades_per_day` | tentativas por sessão | — |
| `daily_stop` | stop diário do próprio agente | usa-se no máximo 90% de `daily_loss_limit` (5%) |
| `daily_target` | meta diária; atingida, o agente para até a próxima sessão | — |
| `after_win_mult` / `after_loss_mult` | multiplicam o tamanho após ganho/perda no dia (anti-martingale / martingale) | risco ≤ `max_risk_per_trade`, alavancagem ≤ `max_leverage` |
| `goal_pressure` | aumenta o risco na 2ª metade da sessão quando está atrás da meta | idem |
| `max_hold` | candles máximos na posição | — |

Novas causas de morte: `ruin` (patrimônio abaixo da menor ordem, mesmo no
teto de alavancagem) e `no_goal` (modo `--fitness goal`: nunca dobrou num dia).
Toda sessão fica registrada em `DayResult.day_log`: risco máximo usado,
tentativas, multiplicador de escalada e como terminou (`daily_stop`,
`daily_target`, `max_trades`, `gate_daily_loss`, `ruin`, `session_end`).

## Modos de seleção (`--fitness`)

| Modo | Classifica por | Mata |
|---|---|---|
| `risk` | retorno ÷ queda máxima | liquidação, ruína, queda forte, taxa, sem vantagem, estagnação |
| `goal` | fração de dias com ≥ +100%, melhor dia | liquidação, ruína, nunca dobrar (`no_goal`) |
| `ladder` | degrau alcançado, depois a nota no degrau | liquidação, ruína, queda no degrau acima do limite, as mortes do `risk` e ficar 3× o mínimo de dias sem bater a referência |
| `all` | roda os três e imprime lado a lado | — |

**Escada:** R$100 → R$1k → R$10k → R$100k. Cada janela roda com o capital do
degrau atual. O agente sobe quando cumpre tudo isto no degrau:
- pelo menos `--ladder-min-days` dias de pregão (60);
- `--ladder-min-profitable` de dias positivos (50%);
- nenhuma ruína;
- nota (retorno ÷ queda) acima da referência nos mesmos dias, e positiva;
- queda máxima da curva diária até `--ladder-max-dd` (15%).

Ao subir, as estatísticas do degrau recomeçam do zero e nasce um filho no
degrau inicial.

**Kelly:** no modo `ladder`, o risco por operação é limitado a
`--kelly-fraction` (½) × E[r]/E[r²]. Os resultados r são medidos em R e vêm
só das operações já encerradas pelo agente, em janelas anteriores e na atual
até aquele momento. O encolhimento soma 30 operações fictícias com média 0.
Sem vantagem comprovada, sobra só o risco de sondagem de 0,2%.

**Ressalva:** com R$100, a ordem mínima da corretora pode forçar um risco
maior que o limite de Kelly. O gate sobe a posição até a ordem mínima quando
isso cabe no teto de risco por operação (2%). O `day_log` registra o risco
pedido, não o efetivo.

## Estilos (`--styles`)

O gene `style` define o timeframe, o tipo de ordem de entrada e as
estratégias elegíveis. O Criador pode trocar de estilo (`switch_style`).

| Estilo | Timeframe | Entrada | Estratégias |
|---|---|---|---|
| scalping | 1m | LIMIT (maker) | momentum, bb_revert, vwap_revert, orb, inside_bar, vol_expansion |
| momentum | 15m | mercado | momentum, donchian, bb_breakout, orb, macd, ema_cross |
| mean_reversion | 15m | LIMIT | vwap_revert, bb_revert, rsi_divergence |
| trend_pullback | 15m | mercado | trend_pullback, supertrend, ema_cross |
| session_breakout | 15m | STOP-ENTRY | orb, pivot_breakout |
| news | 15m | mercado | news_event |
| fade | 15m | STOP-ENTRY | expansion_fade, hammer, engulfing |
| grid | 15m | LIMIT | grid (até `levels` acréscimos, alvo na VWAP) |
| expansion | 15m | STOP-ENTRY | vol_expansion, squeeze_release, expansion_fade |

**Genes de saída por estilo:** `stop`, `take_profit`, `trail` (trailing),
`max_hold` (stop por tempo) e `max_trades_per_day`.

**Modelo de execução:**
- **LIMIT:** executa se o candle seguinte abrir além do preço, ou se negociar
  através dele. "Através" significa a mínima estritamente abaixo do preço
  menos o desconto de fila de 2 bps (`queue_haircut`). A ordem expira em
  `order_expiry` candles. Paga maker de 0,02%; o TP de estilos com limit
  também paga maker.
- **STOP-ENTRY:** executa quando o candle negocia além do gatilho (a
  máxima/mínima do candle do sinal), no gatilho mais `slippage`, ou na
  abertura se o candle abrir além. Paga taker de 0,05%.
- **Nunca no mesmo candle:** decide no fechamento do candle i e só executa a
  partir do i+1. Num candle com entrada no meio, só o stop é checado.

**Ressalva do scalping:** sem livro de ofertas nem ticks, a execução por
limit é uma aproximação otimista. Não sabemos a posição na fila, e o candle de
1m não mostra se o preço passou pelo limite antes ou depois do stop. O
desconto de 2 bps corta os toques de raspão, mas numa fila real a taxa de
execução pode ser bem menor, e as execuções que acontecem tendem a ser as
piores (seleção adversa: o preço atravessa o limite justamente quando continua
contra). Na prática, trate os resultados do scalping como teto, não como
estimativa.

## Regimes e meta-controlador ("saber quando")

`regime.py` calcula, só com candles fechados:
- `range_expansion` (amplitude ÷ maior amplitude dos N anteriores) e `range_vs_atr`;
- `atr_pct` (percentil do ATR no histórico recente) e `volume_z`;
- `efficiency_ratio` (força da tendência);
- distância até a máxima/mínima da sessão, faixa de horário e bloco de sessão;
- `gap` (abertura vs. fechamento anterior).

O rótulo de regime é `quiet` / `normal` / `expanding` / `extreme`.

O gene `regime_map` diz, para cada regime, qual estratégia usar ou
`stand_aside` (não fazer nada). O Criador por regras manda para
`stand_aside` o regime em que o agente mais perdeu, e a autópsia enviada à
LLM diz em qual regime vieram as perdas.

**Setups novos:** `vol_expansion`, `expansion_fade` e `squeeze_release`, todos
com entrada STOP-ENTRY.

## Notícias (`--news none|stub|real`, `experiment`)

- `features.attach_features` junta os provedores pelo **fechamento** de cada candle.
- Genes: `news_gate`, `news_follow`, `event_blackout` (usa `news_minutes_to_next_event`),
  `news_size_mult` e `black_weight`. O `black_weight` fica em 0 no braço W,
  porque o Criador e a população o forçam.
- Braços: A (sem notícias), W (confiáveis), WB (+ black_ bruto), WB' (+ black_rw_ ponderado).
  Mesmas sementes em todos. A estatística é a do `trader/blacknews/experiment.py`
  (`run_experiment`, `daytrade_arms`, `DAYTRADE_PAIRS`).
- **Nome da coluna:** o contrato pedia `minutes_to_next_scheduled_event`, mas o
  provedor real chama a coluna de `news_minutes_to_next_event`, que é a usada aqui.

## Proposta adicional

7. **`trader/evolution/creator.py`:** exportar `_post_json` e
   `_extract_json` sem o sublinhado. O `daytrade/creator.py` importa as duas.

## Limitações conhecidas

- **Execução:** stop e liquidação são avaliados pela máxima e mínima do candle,
  sem livro de ofertas. O stop sai exatamente no preço, salvo gap na abertura,
  então não há slippage além do gap.
- **Preço da liquidação:** usa o último preço, não o *mark price* da Binance.
- **Stop com gap:** um gap que atravessa o stop, mas não a liquidação, sai na
  abertura. Na Binance real, uma ordem stop-market se comportaria de forma
  parecida.
- **Posições:** uma posição por símbolo, só no modo one-way.
