"""CLI de notícias.

    python -m trader.news backfill --source gdelt --asset BTC --since 2024-01-01 --until 2024-02-01
    python -m trader.news backfill --source rss
    python -m trader.news calendar --since 2023-01-01 --until 2026-12-31
    python -m trader.news enrich --limit 200
    python -m trader.news features BTC/USDT 15m --exchange binanceusdm
    python -m trader.news reputation --as-of 2024-02-01 --top 10
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import pandas as pd

from trader.domain import parse_timeframe
from trader.news import calendar as cal
from trader.news import rss
from trader.news.audit import shift_sensitivity, truncation_audit
from trader.news.enrich import Enricher
from trader.news.features import SYMBOL_ASSET, build_features
from trader.news.gdelt import QUERIES, Gdelt
from trader.news.pipeline import CALENDAR_FILE, build_reputation, load_calendar, load_items
from trader.news.store import DEFAULT_LATENCY, NewsStore, make_frame
from trader.storage.candles import CandleStore


def main(argv: list[str] | None = None) -> None:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="trader.news")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--latency-min", type=float, default=DEFAULT_LATENCY.total_seconds() / 60)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backfill")
    b.add_argument("--source", choices=["gdelt", "rss"], required=True)
    b.add_argument("--asset", default="BTC", help=f"um de {sorted(QUERIES)} ou vários separados por vírgula")
    b.add_argument("--query", help="busca GDELT própria (substitui a do ativo)")
    b.add_argument("--since")
    b.add_argument("--until")
    b.add_argument("--window-hours", type=float, default=24)

    c = sub.add_parser("calendar")
    c.add_argument("--since", default="2023-01-01")
    c.add_argument("--until", default="2026-12-31")

    e = sub.add_parser("enrich", help="traduz e rotula com a LLM local (com cache)")
    e.add_argument("--limit", type=int, default=200, help="máximo de itens novos mandados à LLM")
    e.add_argument("--model", default="qwen2.5:7b")
    e.add_argument("--llm-url", default="http://localhost:11434/v1")
    e.add_argument("--horizon", default="4h", help="horizonte da reputação usada para priorizar")
    e.add_argument("--history-daily-cap", type=int,
                   help="modo histórico: por dia, só os N itens mais importantes vão para a LLM")
    e.add_argument("--since", help="modo histórico: início")
    e.add_argument("--until", help="modo histórico: fim")

    f = sub.add_parser("features")
    f.add_argument("symbol")
    f.add_argument("timeframe")
    f.add_argument("--exchange", default="binanceusdm")
    f.add_argument("--reputation", action="store_true", help="pondera fontes pela reputação da época")
    f.add_argument("--horizon", default="4h")
    f.add_argument("--since", help="início do trecho de candles (padrão: primeira notícia)")
    f.add_argument("--until", help="fim do trecho de candles (padrão: última notícia)")

    k = sub.add_parser("gkg-backfill", help="histórico pelos arquivos brutos do GDELT GKG (sem API)")
    k.add_argument("--since", default="2020-01-01")
    k.add_argument("--until", help="padrão: agora")
    k.add_argument("--no-earlier", action="store_true", help="não seguir para 2015-02 → since depois")
    k.add_argument("--feeds", default="en,translation")
    k.add_argument("--max-mbps", type=float, default=4.0, help="teto de banda total em MB/s")
    k.add_argument("--workers", type=int, default=2, help="downloads simultâneos (máx. 2)")
    k.add_argument("--max-files", type=int, help="processa no máximo N arquivos (para medir)")
    k.add_argument("--priority", action="store_true",
                   help="ordem: últimos 12 meses, meses de eventos, resto desde 2020, depois 2015-2019")

    ft = sub.add_parser("fulltext", help="lê o texto completo das notícias recentes de nível 1/2 (respeita robots.txt)")
    ft.add_argument("--limit", type=int, default=15, help="máximo de páginas lidas nesta execução")
    ft.add_argument("--model", default="qwen2.5:7b")
    ft.add_argument("--llm-url", default="http://localhost:11434/v1")

    r = sub.add_parser("reputation")
    r.add_argument("--as-of", required=True)
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--horizon", default="4h")

    args = p.parse_args(argv)
    latency = pd.Timedelta(minutes=args.latency_min)
    store = NewsStore(args.data)

    if args.cmd == "gkg-backfill":
        from trader.news import gkg

        tmp = (args.data / "news" / "tmp").resolve()  # nada de C:
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TEMP"] = os.environ["TMP"] = str(tmp)
        since = pd.Timestamp(args.since, tz="UTC")
        until = pd.Timestamp(args.until, tz="UTC") if args.until else pd.Timestamp.now(tz="UTC")
        lists = args.data / "news" / gkg.SUBDIR / "_lists"
        files = []
        for feed in args.feeds.split(","):
            files += gkg.load_file_list(lists, feed, bandwidth=gkg.Bandwidth(args.max_mbps * 1024 * 1024))
        if args.priority:
            todo = gkg.plan_priority(files, until)
        else:
            todo = gkg.plan(files, since, until, then_earlier=not args.no_earlier)
        if args.max_files:
            todo = todo[: args.max_files]
        print(f"{len(todo)} arquivos no plano ({sum(f.size for f in todo) / 1e9:.1f} GB),"
              f" {since:%Y-%m-%d} → {until:%Y-%m-%d}{'' if args.no_earlier else ', depois 2015-02 → ' + f'{since:%Y-%m-%d}'}",
              flush=True)
        stop = args.data / "news" / gkg.SUBDIR / "STOP"
        stop.unlink(missing_ok=True)
        print(f"para parar com segurança: crie o arquivo {stop}", flush=True)
        stats = gkg.run(args.data, todo, max_mbps=args.max_mbps, workers=min(args.workers, 2), stop_file=stop,
                        log=lambda m: print(f"[{pd.Timestamp.now():%H:%M:%S}] {m}", flush=True))
        print(f"fim: {stats['files']} arquivos, {stats['failed']} falhas, {stats['rows']} linhas,"
              f" {stats['bytes'] / 1e9:.2f} GB em {stats['elapsed'] / 60:.1f} min", flush=True)
        return
    if args.cmd == "backfill":
        if args.source == "rss":
            rows, errors = rss.fetch_all()
            print(f"rss: {len(rows)} itens capturados; novos gravados: {store.write(make_frame(rows))}")
            for feed, err in errors.items():
                print(f"  falhou {feed}: {err}")
            return
        g = Gdelt()
        since, until = pd.Timestamp(args.since, tz="UTC"), pd.Timestamp(args.until, tz="UTC")
        for asset in args.asset.split(","):
            if args.query:
                QUERIES[asset] = args.query
            window = pd.Timedelta(hours=args.window_hours)
            raw_n, kept, new, langs = 0, 0, 0, {}
            for start in pd.date_range(since, until, freq=window, inclusive="left"):
                rows = g.backfill(asset, start, min(start + window, until), window)  # grava a cada janela
                frame = make_frame(rows)
                raw_n, kept, new = raw_n + len(rows), kept + len(frame), new + store.write(frame)
                for lang, n in frame["original_lang"].value_counts().items():
                    langs[lang] = langs.get(lang, 0) + int(n)
            print(f"gdelt {asset}: {raw_n} brutos, {kept} de fontes da lista, {new} novos"
                  f" ({g.requests} requisições até agora); idiomas: {langs}", flush=True)
    elif args.cmd == "calendar":
        parts = [cal.copom(), cal.fomc(), cal.payroll(args.since, args.until)]
        events = pd.concat(parts, ignore_index=True)
        lo, hi = pd.Timestamp(args.since, tz="UTC"), pd.Timestamp(args.until, tz="UTC")
        events = events[(events["scheduled_at"] >= lo) & (events["scheduled_at"] <= hi)].sort_values("scheduled_at")
        path = args.data / CALENDAR_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(path, index=False)
        print(events.groupby("event").size().to_dict(), "->", path)
    elif args.cmd == "fulltext":
        from trader.news.fulltext import fetch_and_label
        from trader.news.pipeline import read_items

        recent = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)  # o texto completo só é lido para notícias de até 24h
        stats = fetch_and_label(args.data, read_items(args.data, since=recent),
                                Enricher(args.data, base_url=args.llm_url, model=args.model),
                                limit=args.limit)
        print("texto completo:", stats)
    elif args.cmd == "enrich":
        enricher = Enricher(args.data, base_url=args.llm_url, model=args.model)
        if args.history_daily_cap:
            from trader.news.pipeline import select_for_llm

            base = load_items(args.data, enricher, latency)
            if args.since:
                base = base[base["visible_at"] >= pd.Timestamp(args.since, tz="UTC")]
            if args.until:
                base = base[base["visible_at"] < pd.Timestamp(args.until, tz="UTC")]
            rep = build_reputation(args.data, base, pd.Timedelta(args.horizon))
            chosen = select_for_llm(base, args.history_daily_cap, rep).head(args.limit)
            print(f"histórico: {len(chosen)} de {len(base)} itens vão para a LLM"
                  f" (tier1={int((chosen['tier'] == 1).sum())}); o resto fica com o tom do GDELT/léxico")
            enricher.enrich(chosen, use_llm=True)
            print(f"chamadas LLM: {enricher.llm_calls}, falhas: {enricher.llm_failures}")
            return
        items = store.read_all()
        pending = items[[enricher.key(t) not in enricher.cache for t in items["original_text"]]]
        if len(pending) > args.limit:
            # reputação para priorizar: bastam os últimos meses (não o GDELT inteiro)
            base = load_items(args.data, enricher, latency, since=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=120))
            rep = build_reputation(args.data, base, pd.Timedelta(args.horizon))
            chosen = rep.priority(pd.Timestamp.now(tz="UTC"), pending.reset_index(drop=True), args.limit, random.Random(0))
            pending = pending.reset_index(drop=True).loc[chosen]
            print(f"priorizados {len(pending)} itens por reputação (20% exploração)")
        enricher.enrich(pending, use_llm=True)
        spi = enricher.seconds_per_item
        print(f"chamadas LLM: {enricher.llm_calls}, falhas: {enricher.llm_failures}, cache: {len(enricher.cache)} itens")
        if spi:
            avg = sum(spi) / len(spi)
            batch = avg * enricher.batch_size  # ao vivo, um item espera o lote inteiro
            print(f"latência medida: {avg:.2f} s por item, ~{batch:.0f} s por lote de {enricher.batch_size}")
            if batch > latency.total_seconds():
                print(f"  ATENÇÃO: ao vivo, traduzir leva mais que a latência configurada ({latency}).")
    elif args.cmd == "features":
        enricher = Enricher(args.data)
        items = load_items(args.data, enricher, latency)
        candles = CandleStore(args.data).read(args.exchange, args.symbol, args.timeframe)
        if items.empty or candles.empty:
            raise SystemExit("faltam notícias ou candles")
        lo = pd.Timestamp(args.since, tz="UTC") if args.since else items["visible_at"].min().floor("D")
        hi = pd.Timestamp(args.until, tz="UTC") if args.until else items["visible_at"].max().ceil("D")
        ts = candles[(candles["ts"] >= lo) & (candles["ts"] < hi)]["ts"].reset_index(drop=True)
        asset = SYMBOL_ASSET.get(args.symbol, args.symbol)
        calendar = load_calendar(args.data)
        rep = build_reputation(args.data, items, pd.Timedelta(args.horizon)) if args.reputation else None

        def build(t, it):
            return build_features(t, args.timeframe, it, asset, calendar, rep)

        feats = build(ts, items)
        out = args.data / "news" / "features" / f"{asset}_{args.timeframe}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(out)
        print(f"{len(feats)} candles, {len(items)} itens ({(items['label_source'] == 'llm').sum()} rotulados pela LLM) -> {out}")
        busiest = feats["news_count_4h"].idxmax()
        print("exemplo (candle com mais notícias):")
        print(feats.loc[busiest].to_string())
        tf = parse_timeframe(args.timeframe)
        leaks = truncation_audit(build, ts, items, lambda t: t + tf, samples=30)
        print("auditoria de truncamento:", "OK, nenhuma diferença" if leaks.empty else f"{len(leaks)} candles divergentes!")
        sens = shift_sensitivity(build, ts, items, pd.Timedelta(minutes=5))
        print("sensibilidade a +5 min de atraso (fração de candles alterados):")
        print(sens["changed_frac"].round(3).to_string())
    elif args.cmd == "reputation":
        items = load_items(args.data, Enricher(args.data), latency)
        rep = build_reputation(args.data, items, pd.Timedelta(args.horizon))
        t = pd.Timestamp(args.as_of, tz="UTC")
        sources = sorted(set(items["domain"]))
        scores = rep.scores(t, sources).sort_values("score", ascending=False)
        hist_dates = list(pd.date_range(items["visible_at"].min().ceil("D"), t, freq="7D"))
        hist = rep.history(hist_dates)
        path = args.data / "news" / "reputation" / "history.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(path, index=False)
        print(f"reputação em {t:%Y-%m-%d} ({len(rep.outcomes)} resultados resolvidos no total; histórico -> {path})")
        print("melhores:\n" + scores.head(args.top).round(3).to_string())
        print("piores:\n" + scores.tail(args.top).round(3).to_string())


if __name__ == "__main__":
    main()
