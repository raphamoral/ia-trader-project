# Notícias point-in-time: integração

## O que existe (`trader/news/`)

| Módulo | Papel |
|---|---|
| `sources.py` | Lista branca de domínios com nível de confiabilidade (1 = agências, bancos centrais, bolsas, grande imprensa financeira; 2 = imprensa cripto estabelecida). Fora da lista = descartado na ingestão. |
| `store.py` | Parquet mensal em `data/news/items/`. Dois relógios por item (`published_at`, `first_seen_at`) e `visible_at` calculado na leitura. `news_until(as_of, assets, lookback)` é a porta de leitura e nunca devolve item com `visible_at > as_of`. |
| `gdelt.py` | Backfill histórico translingual pelo GDELT DOC 2.0. |
| `rss.py` | Captura ao vivo (Valor, CoinDesk, Cointelegraph; InfoMoney falha no Windows por certificado SSL expirado do lado deles). |
| `calendar.py` | Copom (API do BCB), FOMC (página do Fed), payroll (regra da 1ª sexta, marcada como aproximada). |
| `enrich.py` | Uma chamada ao Ollama traduz para inglês e rotula (ativos, sentimento, tipo de evento, confiança) em lotes de 8, com cache em disco por hash. Fallback: léxico. |
| `stories.py` | Dedup entre idiomas: agrupamento online, causal, por semelhança dos títulos em inglês. |
| `reputation.py` | Reputação Beta por fonte e tipo de resultado, point-in-time; API compartilhada com o agente "black". |
| `features.py` | Features por candle no fechamento (contrato da MarketView). |
| `audit.py` | Auditoria de truncamento (deve dar zero) e sensibilidade à latência. |

## Contrato para os motores: `news_features`

```python
from trader.news.features import news_features
feats = news_features("BTC/USDT", "5m", index)   # index: DatetimeIndex UTC com os instantes de decisão
```

- **Causal:** a linha `t` usa só itens com `visible_at <= t`. Passe o fechamento do candle (o instante da decisão); passar a abertura também é seguro, só mais conservador.
- **Mesmo índice** de entrada e todas as colunas começam com `news_`. `float32`, sem NaN: "nenhum evento" vira 7 dias (10080 min).
- **Grades:** intraday (5m, 15m, ...) usa as janelas 1h/4h/24h; `1d` (tickers da B3) usa 24h/3d/7d. Os aliases `news_count`, `news_sentiment`, `news_confidence` e `news_spread` usam a janela principal (4h intraday, 3d diária).
- **Colunas por janela `w`:** `news_count_w`, `news_sentiment_w` (ponderado por confiança x nível x reputação), `news_confidence_w`, `news_spread_w` (itens por história), `news_stories_w`, `news_tier1_w`, `news_langs_w`, `news_sources_w`.
- **Outras colunas:** `news_shock_1h`, `news_event_<tipo>_minutes` (recência de cada tipo de evento: monetary_policy, macro_data, earnings, regulation, hack_security, ...), `news_minutes_to_next_event` e `news_minutes_since_last_event` (calendário macro do ativo).
- **Ativos:** `BTC/USDT`→BTC, `BOVA11`→IBOV, `PETR4`, `VALE3`, `USDBRL`. Outros tickers devolvem zeros com o calendário.
- **Desempenho:** somas acumuladas por ativo, cacheadas em processo (o cache invalida quando os arquivos de notícias mudam), servidas por `searchsorted`, que é uma junção por instante. Medido: 394 mil linhas de 5m do BTC em 1,6 s (2,6 s na primeira chamada), ~70 MB.
- **Reputação no caminho rápido:** o peso de cada item é congelado na reputação do início da semana em que ele ficou visível (resultados resolvidos antes disso). Isso permite somas acumuladas. O limite de fatia por fonte (`cap_shares`) só vale no caminho detalhado (`build_features`), não neste.
- **Parâmetros opcionais** (só por nome): `data=`, `latency=` (padrão 5 min) e `use_reputation=` (padrão True).

## Reputação (CLI)

```
python -m trader.news reputation --as-of 2024-02-01 --top 10 --horizon 4h
```
Grava a série temporal em `data/news/reputation/history.parquet` e mostra as melhores e as piores fontes naquela data.

## Regras de visibilidade

- Backfill: `visible_at = published_at + latência` (padrão 5 min, `--latency-min`). No GDELT, `published_at` é o `seendate` (varredura a cada 15 min), que é conservador.
- Ao vivo: `visible_at = max(published_at + latência, first_seen_at)`. Não dá para ter lido antes de capturar, e relógio de fonte adiantado é cortado para o horário da captura.
- Tradução e rótulos **não** atrasam a visibilidade no backtest. Ao vivo, o tempo de tradução é medido (`translation_seconds` por item). O CLI `enrich` avisa se esse tempo passar da latência configurada; nesse caso, aumente `--latency-min` para o backtest ser honesto.

## Idioma-pivô: inglês

- O Qwen2.5 e os léxicos financeiros são mais fortes em inglês.
- A maioria das fontes de nível 1 publica em inglês.
- O próprio GDELT indexa por tradução para o inglês.

Cada item guarda `original_text`, `original_lang`, `translated_text`, `translator` e `translated_at`.

## Mudanças propostas no código compartilhado (NÃO aplicadas)

1. **`trader/evolution/creator.py`**: tornar públicos `_post_json` e `_extract_json` (renomear para `post_json` e `extract_json`). `enrich.py` hoje importa os nomes privados.
2. **`Pipfile`**: nada novo. O pipeline usa só stdlib, pandas e pyarrow.
3. **Motor com notícias**: a `MarketView` ganharia um campo opcional `news: pd.DataFrame | None` com as features do candle, montado pelo backtest a partir de `data/news/features/{ativo}_{tf}.parquet`, alinhado por `ts`. O backtest de `trader/backtest.py` monta a view em `df.iloc[lo:i+1]`, então bastaria anexar `features.iloc[lo:i+1]`, com o mesmo índice e a mesma causalidade.
4. **Genoma**: genes de filtro por notícia, por exemplo `news_block_minutes_before_event` (não entrar a N minutos de Copom ou FOMC), `min_sentiment_4h` e `shock_exit` (zerar se `shock_1h > 0` contra a posição). Com isso a evolução testa se as notícias ajudam. O Criador com LLM receberia essas features na autópsia (ex.: "morreu 30 min depois do FOMC").
5. **Auditoria na evolução**: rodar `truncation_audit` antes de cada campanha, e `shift_sensitivity` com +5, +15 e +60 min no campeão. Se a vantagem some com +15 min, ela depende de uma velocidade que não teremos ao vivo.
6. **Agente "black"**: usar `reputation.Reputation` com `tiers={canal: BLACK_TIER}`. `story_outcomes` já gera confirmação e pontualidade contra o nível 1. Para retratações, uma coluna `retracted=True` conta como falha de confirmação com peso 3.

## Limitações conhecidas

- **Cobertura do GDELT**: no máximo 250 artigos por requisição (janelas de 1 dia perdem artigos em dias quentes). O limite real de taxa é mais apertado que o documentado (uma requisição a cada 5 s, com 429 frequente); o cliente usa espera crescente. Histórico do DOC 2.0: os últimos anos (a partir de 2017).
- **Títulos, não corpo**: o GDELT DOC devolve só título e URL. Sentimento a partir do título é mais ruidoso.
- **Relógios**: o `seendate` do GDELT pode vir horas depois da publicação real em sites pequenos. Para fontes de nível 1 costuma ficar em até 15 min.
- **RSS**: só os itens recentes. Serve para acumular histórico próprio daqui para frente.
- **CPI**: sem fonte gratuita automatizável verificada (o BLS devolve 403 para acesso automatizado). Payroll é aproximação pela regra da 1ª sexta do mês.
- **Lista branca**: montada à mão, com imprensa asiática e europeia de nível 1 e uma seleção cripto de nível 2. Precisa de revisão periódica; a reputação aprendida ajusta o peso dentro da lista.
