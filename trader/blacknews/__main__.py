"""Linha de comando do agente black.

    python -m trader.blacknews sources
    python -m trader.blacknews backfill --archive-limit 300
    python -m trader.blacknews enrich --limit 120
    python -m trader.blacknews leadlag --since 2026-09-22 --until 2026-09-30 [--verify-calls 3]
    python -m trader.blacknews features BTC/USDT 15m --since 2026-09-25 --until 2026-09-29
    python -m trader.blacknews experiment-demo --seeds 10

Dados em data/news_black/ (nunca em data/news/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trader.blacknews import experiment, leadlag, sources
from trader.blacknews.analysis import BLACK_ROOT, DATA, analyze
from trader.blacknews.api import SYMBOL_ASSET, black_features_from
from trader.blacknews.enrich import enrich
from trader.blacknews.reputation import promotion_candidates
from trader.blacknews.schema import BlackStore
# ativo → candles para medir a janela de preço (mais fino primeiro) e taxa ida e volta
PRICES = {
    "BTC": ([("binanceusdm", "BTC/USDT", "5m"), ("binance", "BTC/USDT", "4h")], 0.001),
    "IBOV": ([("b3", "BOVA11", "1d")], 0.0005),
    "PETR4": ([("b3", "PETR4", "1d")], 0.0005),
    "VALE3": ([("b3", "VALE3", "1d")], 0.0005),
}


def _prices(asset: str) -> pd.DataFrame | None:
    from trader.storage.candles import CandleStore

    store = CandleStore(DATA)
    for exchange, symbol, tf in PRICES.get(asset, ([], 0))[0]:
        c = store.read(exchange, symbol, tf)
        if len(c):
            return c
    return None


def _ts(value: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(value, tz="UTC") if value else None


def main(argv: list[str] | None = None) -> None:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="trader.blacknews")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sources", help="fontes disponíveis e excluídas")
    b = sub.add_parser("backfill", help="coleta o que as fontes oferecem agora (inclui histórico curto)")
    b.add_argument("--archive-limit", type=int, default=200, help="threads arquivadas do 4chan (1 req/s)")
    b.add_argument("--only", default="", help="ids de fonte separados por vírgula")
    e = sub.add_parser("enrich", help="traduz e rotula (LLM local, com cache)")
    e.add_argument("--limit", type=int, default=120, help="itens novos mandados à LLM nesta execução")
    e.add_argument("--no-llm", action="store_true")
    ll = sub.add_parser("leadlag", help="black chega antes ou depois das confiáveis?")
    ll.add_argument("--since")
    ll.add_argument("--until")
    ll.add_argument("--verify-calls", type=int, default=0, help="chamadas à LLM para confirmar pares duvidosos")
    f = sub.add_parser("features", help="features black_ numa grade de candles")
    f.add_argument("symbol")
    f.add_argument("timeframe")
    f.add_argument("--since", required=True)
    f.add_argument("--until", required=True)
    sb = sub.add_parser("sec-backfill", help="monitor de segurança: coleta das fontes aprovadas")
    se = sub.add_parser("sec-extract", help="monitor de segurança: extração de incidentes (LLM local, cache)")
    se.add_argument("--max-calls", type=int, default=15)
    sl = sub.add_parser("sec-leadlag", help="monitor de segurança: incidentes, profundidade e lead/lag")
    ab = sub.add_parser("alt-backfill", help="dados alternativos: coleta pelas APIs públicas aprovadas")
    ab.add_argument("--families", default="fng,wiki,dev,cvm", help="deriv é longo: use --deriv-since")
    ab.add_argument("--deriv-since", default=None, help="baixa métricas de futuros diárias a partir desta data")
    ab.add_argument("--cvm-years", default="2018-2026")
    sub.add_parser("alt-report", help="dados alternativos: profundidade por fonte e fontes recusadas")
    x = sub.add_parser("experiment-demo", help="harness A/B/B'/C com avaliação sintética (stub)")
    x.add_argument("--seeds", type=int, default=10)
    args = p.parse_args(argv)

    if args.cmd == "sources":
        for src in sources.default_sources():
            print(f"  {src.source_id:28s} {src.source_kind:6s} {getattr(src, 'url', getattr(src, 'base', ''))}")
        print("excluídas:")
        for name, why in sources.EXCLUDED.items():
            print(f"  {name:28s} {why}")

    elif args.cmd == "backfill":
        client = sources.PoliteClient()
        now = pd.Timestamp.now(tz="UTC")
        wanted = set(filter(None, args.only.split(",")))
        store = BlackStore(BLACK_ROOT)
        for src in sources.default_sources(args.archive_limit):
            if wanted and src.source_id not in wanted:
                continue
            try:
                items = src.fetch(client, now)
            except sources.SourceDisabled as exc:
                print(f"  {src.source_id}: desligada ({exc})")
                continue
            except Exception as exc:  # uma fonte fora do ar não derruba as outras
                print(f"  {src.source_id}: erro {type(exc).__name__}: {exc}")
                continue
            added = store.write(items)
            span = f"{items['published_at'].min():%Y-%m-%d %H:%M} → {items['published_at'].max():%Y-%m-%d %H:%M}" if len(items) else "-"
            print(f"  {src.source_id}: {len(items)} itens ({added} novos), {span}")

    elif args.cmd == "enrich":
        store = BlackStore(BLACK_ROOT)
        items = store.all()
        todo = items["enriched_by"].isna() | (items["enriched_by"] == "lexicon")
        pending = items[todo].copy()
        # orçamento de LLM: primeiro o que precisa de tradução, depois o mais recente
        from trader.blacknews.enrich import detect_lang

        pending["_en"] = [detect_lang(t or "") == "en" for t in pending["text"]]
        pending = pending.sort_values(["_en", "visible_at"], ascending=[True, False]).drop(columns="_en")
        done, enricher = enrich(pending, BLACK_ROOT, use_llm=not args.no_llm, limit=args.limit)
        merged = pd.concat([items[~todo], done], ignore_index=True)
        store.replace(merged)
        counts = merged["enriched_by"].value_counts().to_dict()
        langs = merged["original_lang"].value_counts().head(8).to_dict()
        secs = enricher.seconds_per_item if enricher else []
        print(f"rótulos: {counts} | línguas: {langs}")
        if enricher:
            per = f"{sum(secs) / len(secs):.2f}s/item" if secs else "-"
            print(f"LLM: {enricher.llm_calls} chamadas, {enricher.llm_failures} falhas, latência de tradução {per}")

    elif args.cmd == "leadlag":
        since, until = _ts(args.since), _ts(args.until)
        a = analyze(since, until, args.verify_calls)
        if a is None:
            raise SystemExit("sem itens black enriquecidos no período; rode backfill e enrich")
        scored, trusted, matches, lags, outs, rep, data_end = (
            a.scored, a.trusted, a.matches, a.lags, a.outcomes, a.reputation, a.data_end)
        edges = leadlag.edge_windows(lags, _prices, lambda a: PRICES[a][1])
        sources_list = sorted(scored["source"].unique())
        scores = rep.scores(data_end, sources_list)
        repdf = pd.DataFrame({"source": sources_list, "mean": scores["score"].to_numpy()})
        table = leadlag.source_table(lags, repdf, edges)
        print(f"black: {len(scored)} itens | confiáveis: {len(trusted)} | pares casados: {len(matches)}")
        print(f"status: {lags['status'].value_counts().to_dict()}")
        with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 60):
            cols = ["source", "kind", "items", "confirmed", "leads", "echoes", "unconfirmed", "median_lag_min",
                    "pct_first", "echo_dominated", "reputation", "rep_bucket"] + [c for c in ("edge_n", "edge_net_median") if c in table]
            print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        if len(table):
            by_bucket = table.groupby("rep_bucket", observed=True)[["leads", "echoes", "unconfirmed"]].sum()
            print("por faixa de reputação:\n" + by_bucket.to_string())
        if len(edges):
            e = edges["net"].dropna()
            print(f"janela de preço (leads com direção): n={len(e)}, líquido mediano {e.median():+.3%}, "
                  f"médio {e.mean():+.3%}, positivos {(e > 0).mean():.0%}")
        else:
            print("janela de preço: nenhum lead com preço disponível")
        promo = promotion_candidates(rep, outs, data_end)
        print(f"candidatas a promoção (só relatório): {promo['source'].tolist() or 'nenhuma'}")
        if lags["status"].isin(["lead", "echo"]).sum() < 30:
            print("AVISO: menos de 30 histórias casadas; amostra pequena demais para conclusões.")

    elif args.cmd == "features":
        from trader.domain import parse_timeframe

        since, until = _ts(args.since), _ts(args.until)
        a = analyze(None, until)
        grid = pd.date_range(since, until, freq=parse_timeframe(args.timeframe), tz="UTC")
        asset = SYMBOL_ASSET.get(args.symbol, args.symbol)
        feats = pd.concat(
            [black_features_from(a, asset, args.timeframe, grid, weighted=w) for w in (False, True)], axis=1
        ).rename_axis("ts").reset_index()
        out = BLACK_ROOT / "features" / f"{asset}_{args.timeframe}_{since:%Y%m%d}_{until:%Y%m%d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(out, index=False)
        busy = feats.sort_values("black_count", ascending=False).head(1).T
        print(f"{len(feats)} linhas → {out}\nexemplo (instante com mais itens):\n{busy.to_string()}")

    elif args.cmd == "sec-backfill":
        from trader.blacknews import security

        store = security.SecurityStore(BLACK_ROOT)
        now = pd.Timestamp.now(tz="UTC")
        for sid, res in security.fetch_all(sources.PoliteClient(), now).items():
            if isinstance(res, str):
                print(f"  {sid}: {res}")
                continue
            added = store.write(res)
            span = f"{res['published_at'].min():%Y-%m-%d} → {res['published_at'].max():%Y-%m-%d}" if len(res) else "-"
            print(f"  {sid}: {len(res)} itens ({added} novos), {span}")

    elif args.cmd == "sec-extract":
        from trader.blacknews import security

        store = security.SecurityStore(BLACK_ROOT)
        items = store.all()
        ex = security.Extractor(security.ExtractCache(BLACK_ROOT / "security" / "extract_cache.sqlite"), max_calls=args.max_calls)
        # orçamento de LLM: itens mais recentes primeiro (os estruturados do DeFiLlama não precisam)
        items = items.sort_values("published_at", ascending=False).reset_index(drop=True)
        out = ex.extract(items)
        store.replace(out)
        print(f"extração: {out['extracted_by'].value_counts().to_dict()} | tipos: {out['incident_type'].value_counts().to_dict()}")
        print(f"LLM: {ex.calls} chamadas, {ex.failures} falhas")

    elif args.cmd == "sec-leadlag":
        from trader.blacknews import security
        from trader.blacknews.analysis import analyze_security

        sec = analyze_security(DATA)
        if sec is None:
            raise SystemExit("sem itens; rode sec-backfill e sec-extract")
        items = sec.items
        print("profundidade por fonte:")
        for sid, g in items.groupby("source_id"):
            inc = g["incident_id"].notna().sum()
            print(f"  {sid:18s} {len(g):5d} itens, {inc:5d} de incidente, {g['published_at'].min():%Y-%m-%d} → "
                  f"{g['published_at'].max():%Y-%m-%d}, carimbo: {','.join(sorted(g['ts_precision'].unique()))}")
        table = security.incidents_table(items)
        print(f"incidentes deduplicados: {len(table)} (multi-fonte: {(table['sources'] > 1).sum()})")
        lags = sec.lags
        lt = leadlag.source_table(lags)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            cols = ["source", "kind", "items", "confirmed", "leads", "echoes", "unconfirmed", "median_lag_min", "pct_first", "echo_dominated"]
            print(lt[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        conf = lags[lags["status"].isin(["lead", "echo"])].merge(items[["id", "entity"]], on="id")
        if len(conf):
            print("incidentes casados com notícia confiável:")
            for r in conf.sort_values("visible_at").itertuples():
                print(f"  {r.visible_at:%Y-%m-%d %H:%M} {r.source:22s} {r.entity:20s} {r.status:5s} {r.lag_min:+.0f} min")
        print(f"status: {lags['status'].value_counts().to_dict()}")
        if lags["status"].isin(["lead", "echo"]).sum() < 30:
            print("AVISO: menos de 30 incidentes casados com notícia confiável; amostra pequena demais para conclusões.")

    elif args.cmd == "alt-backfill":
        from trader.blacknews import alt

        client, store = sources.PoliteClient(), alt.AltStore(BLACK_ROOT)
        fams = set(filter(None, args.families.split(",")))
        jobs = {"fng": lambda: alt.fetch_fng(client), "wiki": lambda: alt.fetch_wiki(client),
                "dev": lambda: alt.fetch_dev(client)}
        a, b = (int(x) for x in args.cvm_years.split("-"))
        jobs["cvm"] = lambda: alt.fetch_cvm(client, list(range(a, b + 1)))
        for fam in [f for f in jobs if f in fams]:
            try:
                df = jobs[fam]()
            except alt.AltSourceNotAllowed as exc:
                print(f"  {fam}: recusada ({exc})")
                continue
            except Exception as exc:
                print(f"  {fam}: erro {type(exc).__name__}: {exc}")
                continue
            print(f"  {fam}: {len(df)} linhas ({store.write(fam, df)} novas)")
        if args.deriv_since:
            done = store.read("deriv")
            have = set(done["ts"].dt.floor("D")) if len(done) else set()
            days = [d for d in pd.date_range(pd.Timestamp(args.deriv_since, tz="UTC"),
                                             pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1), freq="D")
                    if d not in have]
            added = 0
            for k, day in enumerate(days):
                added += store.write("deriv", alt.fetch_deriv_day(client, day))
                if k % 50 == 0:
                    print(f"  deriv: {k}/{len(days)} dias, +{added} linhas", flush=True)
            print(f"  deriv: {len(days)} dias baixados, +{added} linhas")

    elif args.cmd == "alt-report":
        from trader.blacknews import alt

        store = alt.AltStore(BLACK_ROOT)
        for name, p in alt.POLICIES.items():
            if not p.allowed:
                print(f"  {name:15s} RECUSADA: {p.reason}")
                continue
            df = store.read("deriv" if name == "deriv_api" else name)
            if name == "deriv_api":
                continue
            if df.empty:
                print(f"  {name:15s} sem dados ainda (profundidade disponível: {p.depth})")
                continue
            print(f"  {name:15s} {len(df):8d} linhas, {df['key'].nunique():4d} chaves, "
                  f"{df['ts'].min():%Y-%m-%d} → {df['ts'].max():%Y-%m-%d} (disponível: {p.depth})")

    elif args.cmd == "experiment-demo":
        print(experiment.run_experiment(experiment.stub_evaluate, experiment.DEMO_COLUMNS, list(range(args.seeds))).text())


if __name__ == "__main__":
    main()
