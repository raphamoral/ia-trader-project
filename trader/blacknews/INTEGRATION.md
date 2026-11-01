# Agente black de notícias — integração

Fontes não confiáveis (redes sociais, blogs anônimos, sites de conteúdo em massa),
**estritamente separadas** do agente confiável (`trader.news`):

- dados só em `data/news_black/`;
- features só com prefixo `black_` (`black_rw_` = ponderadas pela reputação);
- nenhuma fonte black é promovida automaticamente ao conjunto confiável:
  `promotion_candidates` só reporta. É a separação que mantém válido o experimento.

## O que é reaproveitado do `trader.news` (só leitura, nada editado)

| Peça | De onde | Como |
|---|---|---|
| Regra de visibilidade | `trader.news.store.visible_at` | ao vivo: `max(published_at + 5 min, first_seen_at)`; histórico (`backfilled`): `published_at + 5 min` |
| Tradução + rótulos (pivô inglês) | `trader.news.enrich.Enricher` | cache em `data/news_black/news/cache/labels.jsonl` (separado do confiável) |
| Tokens de história | `trader.news.stories.tokens` | casamento black ↔ confiável |
| Modelo de reputação | `trader.news.reputation.Reputation`, `BLACK_TIER` | prior Beta(1,4), decaimento, `resolved_at < t`, piso de exploração, trava de rajadas |
| Itens confiáveis | `trader.news.pipeline.load_items` | só rótulos já em cache; nunca chama a LLM nem escreve |

Específico daqui: `original_lang` por heurística (as fontes black não informam a língua),
`stance`, `is_claim` (afirmação verificável), spam/robôs, lead/lag e as regras de crédito da reputação.

## Fontes (verificado em 2026-09-29)

| Fonte | Tipo | Situação | Limites | Profundidade histórica |
|---|---|---|---|---|
| 4chan /biz/ (API JSON oficial) | social | **ativa** | ≤1 req/s, If-Modified-Since, atribuição; **uso comercial proibido** | catálogo ao vivo + arquivo ≈ 9–10 dias |
| Medium (RSS por tag) | blog | ativa | feed público; emissor = autor | só os ~10 posts mais recentes por tag |
| Substack (RSS por publicação) | blog | suportado (`RSSFeed`), nenhuma publicação configurada | — | ~20 posts |
| coingape, zycrypto, cryptopolitan, u.today, criptonoticias (es) | farm | ativas | RSS público | 1–6 dias (10–94 itens) |
| bitcoinist, newsbtc, livecoins (pt) | farm | ativas (o erro "certificado expirado" era o repositório de certificados desatualizado do Python do Windows; `sources.py` usa o `certifi` e a verificação continua ligada) | RSS público | 1–2 dias |
| Reddit (API oficial) | social | conector pronto, **desligado** | exige app OAuth aprovado manualmente (desde 2025), não comercial, 100 req/min, termos proíbem treinar IA | só `new` (~100 posts por sub) |
| Bitcointalk | social | **fora** | feed `.xml` desativado pelo admin | — |
| StockTwits | social | **fora** | sem API pública; Cloudflare | — |
| X / Nitter | social | **fora** | termos do X proíbem raspagem | — |
| Telegram `t.me/s` | social | **fora (decisão do usuário)** | termos não autorizam coleta explicitamente | — |

"farm" = alto volume e pouca curadoria; a lista em `sources.FARM_FEEDS` é editável e não é juízo de veracidade.

## Reputação das fontes black

Regras de crédito (`reputation.black_outcomes`), no esquema `OUTCOME_COLUMNS` do confiável:

- **lead confirmado por nível 1** → `confirmation` ✓ e `timeliness` ✓ com bônus de antecedência (até +0,5): o sinal positivo principal;
- lead confirmado só por nível 2 → peso 0,5;
- **eco** (chegou depois) → `confirmation` ✓ com peso 0,1 e `timeliness` ✗: crédito de pontualidade só para leads;
- afirmação sem confirmação em 48 h → `confirmation` ✗ no fim do horizonte;
- guardas: um crédito por (fonte, história), spam pesa metade, cópia de texto não ganha crédito;
- direção de preço fica de fora de propósito (o confiável usa; aqui o sinal é a confirmação).

`reputation.promotion_candidates`: nota ≥ 0,6, ≥ 5 leads em ≥ 3 dias e < 50% de ecos → **só relatório**.

## Lead/lag (`python -m trader.blacknews leadlag`)

`lag = visible_at(black) − primeiro visible_at confiável da mesma história` (negativo = black chegou antes).
Casamento: tokens em inglês + ativo em comum + janela de ±72 h; aceito com Jaccard ≥ 0,30, ou zona
cinzenta (0,12–0,30) confirmada pela LLM (`--verify-calls N`, com cache). A confirmação só conta dentro
de 48 h, igual ao que a reputação saberia na hora. A tabela por fonte traz leads, ecos, sem confirmação,
lag mediano, % primeiro, `echo_dominated` (≥ 70% ecos) e faixa de reputação. A janela de preço mede o
movimento entre o post black e a primeira notícia confiável, na direção do post, **líquido de taxa**
(BTC: futuros 5m, 0,1% ida e volta; B3: diário, 0,05%).

## Como ligar o experimento na evolução real

`experiment.run_experiment(evaluate, feature_columns, seeds)` roda A, B, B' e C com as mesmas sementes.
Falta só o `evaluate` real, que depende de as features de notícia entrarem no genoma (ainda não entraram):

```python
def evaluate(arm: str, columns: list[str], seed: int) -> dict:
    feats = trusted_features.merge(black_features, on="ts", how="left").fillna(0)[["ts", *columns]]
    # 1. a Population recebe `feats` alinhadas à grade de candles (as_of = fechamento do candle);
    # 2. os genes podem referenciar só colunas em `columns` (genoma com máscara por braço);
    # 3. evolução com `seed`; campeão avaliado no holdout.
    return {"score": holdout.score, "goal_hit_rate": goal, "ruin": ruins, "pnl": holdout_pnl_series}

report = run_experiment(evaluate, list(feats.columns), seeds=list(range(10)),
                        suspicious=suspicious_timestamps(black_items))
print(report.text())
```

### API de features para a evolução

```python
from trader.blacknews.api import black_features, columns
fb_u = black_features("BTC/USDT", "15m", decision_index, weighted=False)  # black_* (braço WB)
fb_w = black_features("BTC/USDT", "5m",  decision_index, weighted=True)   # black_rw_* (braço WB')
```

- Saída com **o mesmo índice** de entrada. Passe como índice o **instante da decisão**: o fechamento do
  candle (`ts + timeframe`), não a abertura.
- Causal: linha t usa só itens com `visible_at <= t`. Reputação e líderes usam só desfechos resolvidos
  antes do dia de t. `test_black_features_are_causal_under_truncation` confere isso truncando os
  dados em três cortes.
- Colunas `weighted=False`: black_count, black_count_z, black_mean_sent, black_bull_share,
  black_bear_share, black_spam_share, black_dup_share, black_lang_spread, black_source_spread,
  black_confirmed_rate, black_lead_minutes_median.
- Colunas `weighted=True`: black_rw_count, black_rw_mean_sent, **black_rw_lead_count** e
  **black_rw_lead_score**. As duas `lead` são a soma de nota × sentimento só de fontes que, até o dia
  anterior, tinham ≥1 lead resolvido, leads/(leads+ecos) ≥ 50% e nota acima da priori black. Ecos ficam
  fora por construção.
- Fora do período com dados black, tudo vale 0. A análise é carregada uma vez por processo
  (`lru_cache`), então chamar para 5m e para 15m não repete o trabalho.
- Memória: a análise inteira fica em torno de dezenas de MB com os dados atuais. Ao usar `ProcessPoolExecutor`,
  calcule as features **no processo principal** e mande só o DataFrame pronto aos workers (no máximo 2),
  para não carregar a análise em cada processo.

### Chamada exata para o day trade (braços A / W / WB / WB')

```python
from trader.blacknews.api import black_features, columns
from trader.blacknews.experiment import DAYTRADE_PAIRS, daytrade_arms, run_experiment, suspicious_timestamps
from trader.blacknews.analysis import load_black

idx = candles["ts"] + pd.Timedelta("15min")                       # instante de decisão = fechamento
white = trusted_features_df                                         # do trader.news, mesmo índice, sem prefixo black_
feats = pd.concat([white, black_features("BTC/USDT", "15m", idx, weighted=False),
                   black_features("BTC/USDT", "15m", idx, weighted=True)], axis=1)
arms = daytrade_arms(list(white.columns), columns(False), columns(True))

def evaluate(arm: str, cols: list[str], seed: int) -> dict:
    # DayPopulation com seed=seed, mesmas janelas e holdout em todos os braços;
    # o genoma só pode ler feats[cols] (A: nenhuma coluna de notícia).
    ...
    return {"score": holdout_score,            # retorno / queda máxima no holdout
            "goal_hit_rate": pct_dias_dobrou,  # métrica da meta do usuário
            "ruin": 1.0 if arruinou else 0.0,
            "pnl": pnl_por_instante_do_holdout}  # opcional: habilita o teste de vazamento

report = run_experiment(evaluate, list(feats.columns), seeds=list(range(8)), arms=arms,
                        pairs=DAYTRADE_PAIRS, leak_control="W",
                        suspicious=suspicious_timestamps(load_black()))
print(report.text())
```

`DAYTRADE_PAIRS` compara W×A (notícia confiável ajuda?), WB×W (black ajuda?), WB'×W e WB'×WB
(a ponderação por reputação ajuda?).

Requisitos para o resultado valer: as mesmas janelas/holdout em todos os braços; ≥ 6 sementes (o teste do
sinal não chega a p<0,05 com menos); nenhuma coluna black no braço A (garantido por `arms_for`).
O teste de vazamento compara quanto do ganho de B/B'/C sobre A cai perto de carimbos suspeitos
(publicado depois de coletado, hora cheia vinda do autor, blog/site coletado >1 dia depois e possivelmente editado).

## Pendências e mudanças sugeridas no código compartilhado

1. **Genoma com features de notícia** (dono: evolução): aceitar colunas externas por instante e uma máscara de braço.
2. **Carimbos do confiável:** a primeira coleta do `trader.news` gravou `backfilled=False` em itens publicados até
   2,5 dias antes da coleta (mediana de 8 h), então eles ficam visíveis só na hora da coleta. Isso fazia qualquer
   fonte black parecer "líder" (2 falsos leads de ~13 h na coingape; na verdade eram ecos de ~4 h). O
   `adapters.load_trusted` aplica a mesma convenção das fontes black (coletado >30 min depois = histórico).
   Sugestão ao dono do `trader.news`: marcar `backfilled` na primeira coleta de cada feed.
   `trader.news.store.visible_at` exige a coluna `backfilled`; aqui ela é inferida (coleta > 30 min depois da
   publicação). Se o confiável expuser um coletor ao vivo com a flag explícita, usar a mesma convenção.
3. Léxico do `trader.news.enrich`: a palavra "real" mapeia para USDBRL, o que em inglês coloquial gera falso positivo
   em posts black ("this is real"). Sugestão ao dono: exigir contexto ("real brasileiro", "BRL").
4. Reddit: se o usuário conseguir um app aprovado, exportar `REDDIT_CLIENT_ID/SECRET` e rodar `backfill --only reddit`.
5. Coleta ao vivo contínua (agendada) é o que dá profundidade: o histórico destas fontes é de dias, não de anos.

## Monitor de incidentes de segurança (`security.py`)

Limites fixos: sem Tor, sem login/conta, sem fórum fechado, sem marketplace, nunca abre nem baixa dado
vazado. Guardamos só **título, trecho (≤300 caracteres), link e carimbos** (`make_frame` recusa qualquer
outro campo e `SecurityStore` recusa texto além do trecho). Toda requisição passa por `security.gate`:
fonte não aprovada (ou host diferente do aprovado) é recusada **antes** de qualquer acesso à rede.

| Fonte | Aprovada? | Motivo (verificado em 2026-09-29) | Profundidade | Carimbo |
|---|---|---|---|---|
| DeFiLlama hacks (`api.llama.fi/hacks`) | sim | API pública oficial; termos: uso pessoal não comercial, proibido republicar | **2011 → hoje, ~1.290 incidentes** | só data → visível no fim do dia |
| Rekt News (RSS) | sim | RSS público; robots só bloqueia /api | 7 posts (2025–2026), quase só ensaios | só data |
| SlowMist / PeckShield / CertiK (RSS do Medium) | sim | feed público | 10 posts cada; o do PeckShield parou em 2021 | com hora |
| Status Coinbase / Kraken / Gemini (`history.rss`) | sim | RSS público; robots só bloqueia /api e /embed | ~25 incidentes, 2–5 semanas | hora da **primeira** atualização, tirada do corpo |
| Ahmia (índice onion) | **não** | robots.txt: `Disallow: /search/` e `/*?`, sem API documentada | — | — |
| CertiK API | **não** | robots.txt: `Disallow: /api/` | — | — |
| Status Bitfinex / OKX | **não** | sem resposta na verificação | — | — |

Alertas em tempo real de PeckShield e CertiK saem principalmente no X, que está fora (termos). Por isso a
parte "antes da imprensa" depende das páginas de status das exchanges e da coleta ao vivo contínua.

- **Extração:** Qwen local em lotes de 10, com cache SQLite (`security/extract_cache.sqlite`). Extrai entidade,
  tipo de entidade, tipo de incidente, valor alegado e confiança. Cai para regras se a LLM falhar. Os itens do
  DeFiLlama já vêm estruturados e não usam a LLM.
- **Deduplicação:** mesma entidade normalizada dentro de 3 dias = mesmo incidente, em qualquer fonte. É feita
  online, na ordem de visibilidade (causal).
- **Confirmação:** notícia de nível 1 que cita a entidade (±72 h) ou página de status oficial da própria exchange.
  A feature só a enxerga depois que ela ficou visível.
- **Reputação:** as mesmas regras de crédito das fontes black (lead confirmado ✓, eco sem crédito de pontualidade),
  mas com priori **média** (`tier=None` do `trader.news`, Beta(1; 1,5)).
- **Features** (entram em `black_features` automaticamente):
  - `weighted=False`: `black_sec_incident_count`, `black_sec_max_claimed_usd`,
    `black_sec_minutes_since_incident` (7 dias quando não houve incidente) e `black_sec_confirmed_flag`, todas
    na janela de 24 h;
  - `weighted=True`: `black_rw_sec_incident_score`.
  - Incidentes cripto → BTC. Empresas da B3 citadas (Petrobras, Vale, Itaú…) → o ticker delas.
- **Experimento:** as colunas `black_sec_*` já entram nos braços WB/WB' pelo prefixo. Para medir o monitor
  isoladamente, crie um braço extra:
  `arms["WS"] = [*white, *security.SEC_FEATURES]` e o par `("WS", "W")`.
- **Cuidado com o DeFiLlama em backtest:** a base foi montada depois dos fatos (viés de retrospectiva). Os valores
  são estimativas finais, não o que se sabia no dia. Incidentes que ninguém notou na época também estão lá. Serve
  para "houve um hack grande ontem?", mas não para medir quem noticiou primeiro.

CLI: `python -m trader.blacknews sec-backfill`, `sec-extract --max-calls N`, `sec-leadlag`.
- **Cobertura confiável:** lead/eco só é julgado quando a coleta confiável estava rodando em volta do instante, com
  ≥1 item confiável nas 24 h antes **e** nas 24 h depois (`leadlag.trusted_covered`). Fora disso o status é
  `no_coverage`, sem crédito e sem falha. Hoje a base confiável tem só jan/2024 (GDELT) e 25–29/09/2026, então
  **nenhum incidente pôde ser julgado** (1.312 `no_coverage`). O "lead" de 37 h do DeFiLlama sobre o hack da
  Bitget era artefato dessa falta de cobertura.
- As features contam só hack, exploit, alegação de vazamento, rug pull e parada de saque. Paradas rotineiras das
  páginas de status ("depósitos atrasados") ficam guardadas, mas não entram.


## Dados alternativos (`alt.py`) — `black_alt_<família>_*`

Só APIs e bases públicas oficiais, pelo portão `alt.gate` (fonte recusada ou host diferente = nenhuma requisição).
User-Agent com contato, intervalos respeitados (GitHub 65 s, CVM 10 s pelo Crawl-Delay).

| Família | Fonte | Situação | Profundidade | Visível em |
|---|---|---|---|---|
| fng | alternative.me Fear & Greed | ok (uso livre com atribuição, 60 req/min) | **fev/2018 → hoje**, diário | dia D → D+1 00:00 (hora exata desconhecida) |
| wiki | Wikimedia pageviews (Bitcoin, Cryptocurrency, Binance, Coinbase; pt: Petrobras, Vale, Ibovespa) | ok | **jul/2015 → hoje**, diário | dia D → D+1 06:00 |
| deriv | data.binance.vision (OI, long/short de top traders e geral, taker buy/sell), 5 min | ok | **dez/2021 → hoje** (baixado desde jan/2025; `--deriv-since` estende) | create_time + 5 min |
| deriv_api | fapi.binance.com `futures/data/*` | ok | **só ~30 dias** (por isso a base diária acima) | — |
| dev | GitHub releases (bitcoin, go-ethereum, agave, rippled) | ok, 60 req/h sem autenticação | anos | published_at + 5 min |
| cvm | CVM Dados Abertos, VLMO (negociação de administradores, art. 11 Res. 44 / Instr. 358) | ok | **2018 → hoje**, mensal | **Data_Entrega (protocolo) + 1 dia**; reapresentação só vale após o próprio protocolo |
| — | Whale Alert | **recusada** | sem API gratuita | — |
| — | fluxo de entrada/saída das exchanges | **recusada** | nenhuma fonte pública gratuita oficial | — |
| — | Polymarket / Kalshi | **recusadas** | inacessíveis desta rede; não contornamos (regulador declarou mercados de previsão não autorizados no Brasil em mar/2026) | — |
| — | CoinGecko trending / YouTube | **recusadas** | exigem chave ligada a conta | — |
| — | B3 aluguel de ações | **pendente** | só pelo Boletim Diário via endpoint interno não documentado; API oficial exige cadastro. Baixar manualmente, como o COTAHIST, e escrever o leitor | desde 25/09/2023 |

Features:

- **fng** (cripto): `black_alt_fng_value`, `black_alt_fng_chg_7d`.
- **wiki**:
  - `black_alt_wiki_views_z`: log das visitas contra os 28 dias anteriores;
  - `black_alt_wiki_views_chg_1d`.
- **deriv** (cripto): `black_alt_deriv_oi_chg_1h`, `black_alt_deriv_oi_chg_24h`, `black_alt_deriv_top_ls`,
  `black_alt_deriv_ls`, `black_alt_deriv_taker`, `black_alt_deriv_taker_z`.
- **dev** (cripto): `black_alt_dev_releases_7d`, `black_alt_dev_days_since_release`.
- **cvm** (B3): `black_alt_cvm_net_volume_30d` (R$ líquido dos administradores em formulários protocolados nos
  últimos 30 dias) e `black_alt_cvm_filings_30d`.

```python
from trader.blacknews.api import alt_features
from trader.blacknews.alt import alt_arms, alt_pairs, alt_reputation
fa = alt_features("BTC/USDT", "15m", decision_index)                 # todas as famílias aplicáveis ao ativo
arms = alt_arms(list(white.columns), list(fa.columns))               # W, W+fng, W+wiki, W+deriv, W+dev, W+alt
report = run_experiment(evaluate, [], seeds, arms=arms, pairs=alt_pairs(arms), leak_control="W")
rep = alt_reputation(fa, close_prices)                               # nota por família (direção), point-in-time
```

Liquidações da Binance (`forceOrders`) agora exigem chave: ficaram de fora. A base diária de métricas só existe
para os pares USDT-M.
