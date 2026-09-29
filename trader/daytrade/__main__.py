"""Linha de comando do day trade alavancado (simulação).

    python -m trader.daytrade collect SOL/USDT 15m --since 2023-01-01
    python -m trader.daytrade evolve SOL/USDT 15m --seeds 0,1,2 --fitness all --workers 2
    python -m trader.daytrade evolve SOL/USDT 15m --styles all --seeds 0,1,2
    python -m trader.daytrade experiment SOL/USDT 15m --news stub --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import gc
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from trader.backtest import walk_forward_windows
from trader.daytrade import data
from trader.daytrade.backtest import KellyCap, funding_lookup, goal_metrics, run_daytrade_backtest
from trader.daytrade.creator import DayLLMCreator
from trader.daytrade.evolution import (
    DAY_BASELINE,
    LADDER_BRL,
    DayConfig,
    DayPopulation,
    LadderRules,
    daily_curve_stats,
)
from trader.daytrade.features import attach_features, real_black, real_white, stub_black, stub_white
from trader.daytrade.genome import STYLES
from trader.daytrade.regime import REGIMES
from trader.domain import parse_timeframe
from trader.daytrade.gate import DayTradeLimits
from trader.daytrade.market import brl_to_usdt, feasibility, spec_for
from trader.evolution.fitness import risk_adjusted
from trader.evolution.registry import Registry
from trader.storage.candles import CandleStore

MODES = ("risk", "goal", "ladder")


def main(argv: list[str] | None = None) -> None:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(prog="trader.daytrade")
    parser.add_argument("--data", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)

    col = sub.add_parser("collect", help="candles e funding dos futuros USDT-M")
    col.add_argument("symbol")
    col.add_argument("timeframe")
    col.add_argument("--since", default="2023-01-01")

    evo = sub.add_parser("evolve", help="evolução de day trade alavancado em simulação")
    exp = sub.add_parser("experiment", help="braços A/W/WB/WB' de notícias, mesmas sementes")
    for sp in (evo, exp):
        add_common(sp)
    exp.add_argument("--population-exp", type=int, default=12)

    args = parser.parse_args(argv)
    if args.command == "collect":
        start = pd.Timestamp(args.since, tz="UTC")
        added = data.update_candles(CandleStore(args.data), args.symbol, args.timeframe, start)
        fund = data.update_funding(args.data, args.symbol, start)
        print(f"{args.symbol} {args.timeframe}: +{added} candles; funding: +{fund}")
    elif args.command == "experiment":
        experiment(args)
    else:
        evolve(args)


def add_common(evo: argparse.ArgumentParser) -> None:
    evo.add_argument("symbol")
    evo.add_argument("timeframe")
    evo.add_argument("--window-days", type=int, default=30)
    evo.add_argument("--holdout-days", type=int, default=90)
    evo.add_argument("--population", type=int, default=16)
    evo.add_argument("--slots", type=int, default=32)
    evo.add_argument("--seeds", default="0")
    evo.add_argument("--workers", type=int, default=2)
    evo.add_argument("--max-leverage", type=float, default=3.0, help="teto congelado do gate")
    evo.add_argument("--max-risk", type=float, default=0.02, help="teto congelado de risco por operação")
    evo.add_argument("--daily-loss", type=float, default=0.05, help="teto congelado de perda diária")
    evo.add_argument("--capital-brl", type=float, default=100.0)
    evo.add_argument("--brl-rate", type=float, default=5.5, help="reais por USDT")
    evo.add_argument("--fitness", choices=[*MODES, "all"], default="risk")
    evo.add_argument("--ladder-min-days", type=int, default=60)
    evo.add_argument("--ladder-min-profitable", type=float, default=0.5)
    evo.add_argument("--ladder-max-dd", type=float, default=0.15)
    evo.add_argument("--kelly-fraction", type=float, default=0.5)
    evo.add_argument("--llm", action="store_true")
    evo.add_argument("--llm-url", default="http://localhost:11434/v1")
    evo.add_argument("--llm-model", default="qwen2.5:7b")
    evo.add_argument("--styles", default="any", help="lista separada por vírgula, ou all")
    evo.add_argument("--news", choices=["none", "stub", "real"], default="none")
    evo.add_argument("--evo-start", default=None, help="início da evolução (ex.: 2025-01-01)")


class Setup:
    """Tudo que as rodadas de todos os modos compartilham."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.candles = CandleStore(args.data).read(data.EXCHANGE, args.symbol, args.timeframe)
        if self.candles.empty:
            raise SystemExit(f"sem dados: python -m trader.daytrade collect {args.symbol} {args.timeframe}")
        self.funding = funding_lookup(data.read_funding(args.data, args.symbol))
        self.cash = brl_to_usdt(args.capital_brl, args.brl_rate)
        self.spec = spec_for(args.symbol)
        self.limits = DayTradeLimits(
            max_leverage=args.max_leverage, max_risk_per_trade=args.max_risk, daily_loss_limit=args.daily_loss
        )
        self.ladder = LadderRules(
            min_days=args.ladder_min_days, min_profitable_share=args.ladder_min_profitable,
            max_drawdown=args.ladder_max_dd, kelly=KellyCap(fraction=args.kelly_fraction),
        )
        self.styles = tuple(STYLES) if args.styles == "all" else tuple(args.styles.split(","))
        first, last = self.candles["ts"].iloc[0], self.candles["ts"].iloc[-1]
        # estilos com outro timeframe (scalping 1m) só podem evoluir onde há dados deles
        for tf in {STYLES[s]["timeframe"] for s in self.styles} - {args.timeframe}:
            other = CandleStore(args.data).last_ts(data.EXCHANGE, args.symbol, tf)
            if other is None:
                raise SystemExit(f"sem candles {tf}: python -m trader.daytrade.collect_chunked {args.symbol} {tf} 2024-01-01")
            years = sorted(p.stem for p in (args.data / "candles" / data.EXCHANGE / args.symbol.replace("/", "") / tf).glob("*.parquet"))
            first = max(first, pd.Timestamp(f"{years[0]}-01-01", tz="UTC"))
        if args.evo_start:
            first = max(first, pd.Timestamp(args.evo_start, tz="UTC"))
        self.hold_start = (last - pd.Timedelta(days=args.holdout_days)).normalize()
        self.windows = walk_forward_windows(
            first.normalize() + pd.Timedelta(days=3), self.hold_start, pd.Timedelta(days=args.window_days)
        )
        self.end = last + pd.Timedelta(seconds=1)
        self.hold_data = self.candles[self.candles["ts"] >= self.hold_start - pd.Timedelta(days=5)].reset_index(drop=True)

    def loader(self, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Candles de outro timeframe por intervalo de datas (1m não cabe inteiro na RAM)."""
        return CandleStore(self.args.data).read(data.EXCHANGE, self.args.symbol, timeframe, start, end)

    def providers_for(self, arm: str | None = None) -> list:
        mode = self.args.news
        if mode == "none" or arm == "A":
            return []
        white = stub_white(0) if mode == "stub" else real_white(str(self.args.data))
        if arm in (None, "W"):
            return [white]
        weighted = arm == "WB'"
        black = stub_black(0, weighted) if mode == "stub" else real_black(weighted, str(self.args.data))
        return [white, black]

    def holdout(self, genome, delay: int = 0, kelly: KellyCap | None = None, prior_r=None, providers=None):
        tf = genome.timeframe
        if tf == self.args.timeframe:
            frame = self.hold_data
        else:
            frame = self.loader(tf, self.hold_start - pd.Timedelta(days=5), self.end)
        if providers:
            frame = attach_features(frame, self.args.symbol, tf, providers)
        return run_daytrade_backtest(
            genome, frame, self.args.symbol, tf, self.hold_start, self.end,
            cash=self.cash, limits=self.limits, funding=self.funding, spec=self.spec, delay=delay,
            kelly=kelly, prior_r=prior_r,
        )


def evolve(args: argparse.Namespace) -> None:
    st = Setup(args)
    print("funding:", "histórico real" if st.funding else "constante 0,01%/8h (sem histórico)")
    ok, msg = feasibility(args.symbol, st.cash, args.max_leverage, float(st.candles["close"].iloc[-1]))
    print(f"capital R${args.capital_brl:g} ÷ {args.brl_rate:g} = {st.cash:.2f} USDT | {msg}")
    if not ok:
        raise SystemExit("impossível operar com esse capital e teto; aumente --max-leverage/--capital-brl ou use outro símbolo")
    print(f"teto congelado: {st.limits.max_leverage:g}x, risco/op ≤ {st.limits.max_risk_per_trade:.0%},"
          f" perda diária ≤ {st.limits.daily_loss_limit:.0%} | {len(st.windows)} gerações de {args.window_days}d"
          f" até {st.hold_start:%Y-%m-%d}")

    modes = MODES if args.fitness == "all" else (args.fitness,)
    results = {}
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        for mode in modes:
            results[mode] = run_mode(st, mode, executor)
            gc.collect()
    finally:
        if executor:
            executor.shutdown()
    report(st, results)


def run_mode(st: Setup, mode: str, executor, arm: str | None = None, seeds: list[int] | None = None,
             population: int | None = None) -> dict:
    args = st.args
    providers = st.providers_for(arm)
    has_black = any(getattr(p, "__name__", "").startswith(("stub_black", "black_")) for p in providers)
    config = DayConfig(
        initial=population or args.population, max_slots=args.slots if population is None else population + 8,
        limits=st.limits, fitness=mode, cash=st.cash,
        capital_brl=args.capital_brl, spec=st.spec, brl_rate=args.brl_rate, ladder=st.ladder,
        styles=st.styles, news=bool(providers), black=has_black,
    )
    rows, totals = [], Counter()
    survivors: list = []
    for seed in seeds if seeds is not None else [int(s) for s in args.seeds.split(",")]:
        run = f"{pd.Timestamp.now():%Y%m%d-%H%M%S}-day-{mode}{'-' + arm if arm else ''}-s{seed}"
        registry = Registry(args.data / "evolution" / f"{run}.sqlite")
        creator = None
        if args.llm:
            creator = DayLLMCreator(args.symbol, args.timeframe, llm_context(args, st),
                                    args.llm_url, args.llm_model, graveyard=lambda: registry.graveyard(8))
        pop = DayPopulation(st.candles, args.symbol, args.timeframe, registry, config, st.funding, seed, executor,
                            creator, loader=st.loader, providers=providers)
        pop.seed()
        for a, b in st.windows:
            pop.step(a, b)
        totals.update(pop.totals)
        survivors.extend((a.genome.style, a.genome.regime_map, dict(a.regime_pnl)) for a in pop.alive)
        line = f"[{mode} s{seed}] mortes {registry.cause_counts()}"
        if mode == "ladder":
            line += f" | degraus dos vivos {pop.rung_census()} | promoções {len(pop.promotions)}"
        print(line)
        champ = pop.champion()
        if champ is None:
            rows.append({"seed": seed, "champion": None})
        else:
            kelly = st.ladder.kelly if mode == "ladder" else None
            prior = champ.r_history[-500:] if mode == "ladder" else None
            rows.append({
                "seed": seed, "champion": champ.id, "genome": champ.genome.to_dict(),
                "rung": champ.rung, "days_per_rung": dict(champ.days_per_rung), "rung_stats": champ.rung_stats(),
                "life_trades": champ.life.trades,
                "holdout": st.holdout(champ.genome, kelly=kelly, prior_r=prior, providers=providers),
                "delayed": st.holdout(champ.genome, delay=1, kelly=kelly, prior_r=prior, providers=providers),
                "style": champ.genome.style,
            })
        registry.db.close()
        del pop, registry
        gc.collect()
    return {"rows": rows, "totals": totals, "survivors": survivors}


def report(st: Setup, results: dict) -> None:
    args = st.args
    base = st.holdout(DAY_BASELINE)
    hw = st.candles[st.candles["ts"] >= st.hold_start]
    closes = hw["close"].to_numpy()
    bh_ret = closes[-1] / hw["open"].iloc[0] - 1
    bh_dd = float((1 - closes / np.maximum.accumulate(closes)).max())

    print("\nEVOLUÇÃO (todas as sementes)")
    for mode, res in results.items():
        t = res["totals"]
        extra = ""
        if mode == "ladder":
            extra = " | promoções " + ", ".join(
                f"R${LADDER_BRL[k]:,.0f}: {t.get(f'promotions_to_{k}', 0)}" for k in range(1, len(LADDER_BRL))
            )
        print(f"  {mode:<6} {t['agent_days']:>6} agente-dias, dobraram {t['doubled_days']}, ruínas {t['ruins']},"
              f" liquidações {t['liquidations']}, dias escalados {t['escalated_days']} → ruína"
              f" {t['escalated_ruin']} / dobrou {t['escalated_double']}{extra}")

    print(f"\nHOLDOUT {args.holdout_days}d (nunca visto), capital R${args.capital_brl:g}, os três modos lado a lado")
    print("risco = retorno / queda máxima | meta = dias com ≥ +100% | escada = degrau alcançado na evolução")
    print(f"{'modo':<6} {'sem':>3} {'campeão':>8} {'retorno':>8} {'queda':>6} {'ops':>4} {'liq':>3} {'ruína':>5}"
          f" {'nota':>6} │ {'dias≥2x':>7} {'melhor':>7} {'p/R$1k/dia':>10} │ {'degrau':>9} {'dias/degrau':>18}"
          f" {'%dias+':>6} {'nota degrau':>11}")

    def goal_cols(r):
        g = goal_metrics(r.daily, args.capital_brl)
        dt = "∞" if g["days_to_income"] == float("inf") else f"{g['days_to_income']:.0f}"
        return f"{g['hits']:>3}/{g['days']:<3} {g['best_day']:>+7.1%} {dt:>10}"

    for mode, res in results.items():
        for row in res["rows"]:
            if row["champion"] is None:
                print(f"{mode:<6} {row['seed']:>3} {'—':>8}")
                continue
            r, rs = row["holdout"], row["rung_stats"]
            dpr = ",".join(f"{k}:{v}" for k, v in sorted(row["days_per_rung"].items()))
            rung = f"R${LADDER_BRL[row['rung']]:,.0f}"
            print(f"{mode:<6} {row['seed']:>3} {row['champion']:>8} {r.total_return:>+8.1%} {r.max_drawdown:>6.1%}"
                  f" {r.n_trades:>4} {r.liquidations:>3} {'sim' if r.ruined else 'não':>5} {r.score:>6.2f} │"
                  f" {goal_cols(r)} │ {rung:>9}"
                  f" {dpr:>18} {rs['profitable_share']:>6.0%} {rs['score']:>11.2f}")
    print(f"{'ref':<6} {'':>3} {'':>8} {base.total_return:>+8.1%} {base.max_drawdown:>6.1%} {base.n_trades:>4}"
          f" {base.liquidations:>3} {'sim' if base.ruined else 'não':>5} {base.score:>6.2f} │ {goal_cols(base)} │")
    print(f"{'c/s 1x':<6} {'':>3} {'':>8} {bh_ret:>+8.1%} {bh_dd:>6.1%} {'':>4} {'':>3} {'':>5}"
          f" {risk_adjusted(bh_ret, bh_dd):>6.2f} │")

    print("\nAUDITORIA DE VAZAMENTO (sinais atrasados 1 candle)")
    for mode, res in results.items():
        for row in res["rows"]:
            if row["champion"] is None:
                continue
            r, d = row["holdout"], row["delayed"]
            g = goal_metrics(d.daily, args.capital_brl)
            suspicious = g["geo_daily"] > 0.05 or g["hit_rate"] > 0.02 or d.total_return > 5
            print(f"  {mode:<6} s{row['seed']} {row['champion']}: normal {r.total_return:>+7.1%} → atrasado"
                  f" {d.total_return:>+7.1%} (média geom./dia {g['geo_daily']:+.2%})"
                  f"{'  ⚠ SUSPEITO' if suspicious else '  ok'}")

    for mode, res in results.items():
        style_report(mode, res)
        regime_report(mode, res)

    print("\nCAMPEÕES")
    for mode, res in results.items():
        for row in res["rows"]:
            if row["champion"] is not None:
                print(f"  {mode:<6} s{row['seed']} {row['champion']}: {row['genome']}")


def style_report(mode: str, res: dict) -> None:
    t = res["totals"]
    styles = sorted({k.split(":", 1)[1] for k in t if k.startswith("style_born:")})
    if len(styles) <= 1 and styles == ["any"]:
        return
    alive = Counter(s for s, _, _ in res["survivors"])
    best = {}
    for row in res["rows"]:
        if row["champion"] is not None:
            r = row["holdout"]
            if row["style"] not in best or r.score > best[row["style"]][1]:
                best[row["style"]] = (row["champion"], r.score, r.total_return)
    print(f"\nPOR ESTILO ({mode}) — sobrevivência = vivos no fim ÷ nascidos")
    print(f"{'estilo':<17} {'nascidos':>8} {'vivos':>5} {'sobrev.':>7} {'taxa/bruto':>10} {'fill maker':>10}  melhor no holdout")
    for s_ in styles:
        born = t[f"style_born:{s_}"]
        gross, fees = t[f"style_gross:{s_}"], t[f"style_fees:{s_}"]
        placed, filled = t[f"style_limit_placed:{s_}"], t[f"style_limit_filled:{s_}"]
        share = f"{fees / gross:.0%}" if gross > 0 else "bruto≤0"
        fill = f"{filled / placed:.0%}" if placed else "—"
        b = best.get(s_)
        bs = f"{b[0]} nota {b[1]:.2f} ({b[2]:+.1%})" if b else "—"
        print(f"{s_:<17} {born:>8} {alive[s_]:>5} {alive[s_] / max(born, 1):>7.0%} {share:>10} {fill:>10}  {bs}")


def regime_report(mode: str, res: dict) -> None:
    t = res["totals"]
    bars = {r: t.get(f"regime_bars:{r}", 0) for r in REGIMES}
    total_bars = sum(bars.values())
    if not total_bars:
        return
    pnl = {r: t.get(f"regime_pnl:{r}", 0.0) for r in REGIMES}
    abs_pnl = sum(abs(v) for v in pnl.values()) or 1.0
    print(f"\nPOR REGIME ({mode}) — agente-candles; PnL por regime de entrada; stand_aside = mapa mandou ficar de fora")
    print(f"{'regime':<10} {'tempo':>6} {'PnL (USDT)':>11} {'|PnL| %':>7}  estilos vivos com PnL>0 nele")
    for r in REGIMES:
        winners = Counter(s for s, _, rp in res["survivors"] if rp.get(r, 0) > 0)
        ws = ", ".join(f"{k}:{v}" for k, v in winners.most_common(4)) or "—"
        print(f"{r:<10} {bars[r] / total_bars:>6.1%} {pnl[r]:>+11.2f} {abs(pnl[r]) / abs_pnl:>7.0%}  {ws}")
    aside = t.get("stand_aside_bars", 0)
    maps = [m for _, m, _ in res["survivors"] if m]
    aside_by = Counter(r for m in maps for r, v in m.items() if v == "stand_aside")
    print(f"stand_aside: {aside / total_bars:.1%} dos agente-candles; vivos com mapa: {len(maps)}/{len(res['survivors'])};"
          f" regimes deixados de lado por eles: {dict(aside_by)}")


def experiment(args: argparse.Namespace) -> None:
    """Braços A (sem notícias), W (confiáveis), WB (+ black), WB' (+ black ponderada), mesmas sementes."""
    from trader.blacknews.experiment import DAYTRADE_PAIRS, daytrade_arms, run_experiment

    if args.news == "none":
        raise SystemExit("use --news stub ou --news real")
    st = Setup(args)
    print(f"notícias: {args.news} | {len(st.windows)} gerações de {args.window_days}d até {st.hold_start:%Y-%m-%d}"
          f" | estilos {','.join(st.styles)}")
    probe_idx = pd.DatetimeIndex(st.hold_data["ts"].iloc[:50] + parse_timeframe(args.timeframe))
    cols = {arm: [c for p in st.providers_for(arm) for c in p(args.symbol, args.timeframe, probe_idx).columns]
            for arm in ("W", "WB", "WB'")}
    white = cols["W"]
    arms = daytrade_arms(white, [c for c in cols["WB"] if c.startswith("black_")],
                         [c for c in cols["WB'"] if c.startswith("black_")])
    extra: dict[tuple[str, int], dict] = {}
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None

    def evaluate(arm: str, columns: list[str], seed: int) -> dict:
        res = run_mode(st, args.fitness if args.fitness in MODES else "risk", executor, arm=arm, seeds=[seed],
                       population=args.population_exp)
        row = res["rows"][0]
        if row["champion"] is None:
            extra[(arm, seed)] = {"champion": None}
            return {"score": np.nan, "goal_hit_rate": np.nan, "ruin": np.nan}
        r = row["holdout"]
        g = goal_metrics(r.daily, args.capital_brl)
        ret, dd = daily_curve_stats(list(r.daily))
        rung_ok = (len(r.daily) >= st.ladder.min_days and (r.daily > 0).mean() >= st.ladder.min_profitable_share
                   and dd <= st.ladder.max_drawdown and r.score > 0)
        extra[(arm, seed)] = {"champion": row["champion"], "ret": r.total_return, "dd": r.max_drawdown,
                              "liq": r.liquidations, "ruined": r.ruined, "profitable": float((r.daily > 0).mean()),
                              "rung_ok": rung_ok, "trades": r.n_trades, "black_weight": row["genome"]["black_weight"]}
        pnl = pd.Series(np.diff(np.concatenate([[r.initial_cash], r.equity.to_numpy()])), index=r.equity.index)
        print(f"  [{arm} s{seed}] {row['champion']} nota {r.score:+.2f} retorno {r.total_return:+.1%}"
              f" black_weight {row['genome']['black_weight']:.2f}", flush=True)
        return {"score": r.score, "goal_hit_rate": g["hit_rate"], "ruin": float(r.ruined), "pnl": pnl}

    try:
        seeds = [int(x) for x in args.seeds.split(",")]
        report_ = run_experiment(evaluate, white, seeds, arms=arms, pairs=DAYTRADE_PAIRS, leak_control="W")
    finally:
        if executor:
            executor.shutdown()
    print()
    print(report_.text())
    print("\nPOR BRAÇO (holdout; escada = holdout cumpre os critérios para subir de R$100 a R$1k)")
    print(f"{'braço':<5} {'sem':>3} {'campeão':>8} {'retorno':>8} {'queda':>6} {'ops':>4} {'liq':>3} {'ruína':>5}"
          f" {'%dias+':>6} {'sobe?':>5} {'black_w':>7}")
    for (arm, seed), e in sorted(extra.items(), key=lambda kv: (list(arms).index(kv[0][0]), kv[0][1])):
        if e["champion"] is None:
            print(f"{arm:<5} {seed:>3} {'—':>8}")
            continue
        print(f"{arm:<5} {seed:>3} {e['champion']:>8} {e['ret']:>+8.1%} {e['dd']:>6.1%} {e['trades']:>4} {e['liq']:>3}"
              f" {'sim' if e['ruined'] else 'não':>5} {e['profitable']:>6.0%} {'sim' if e['rung_ok'] else 'não':>5}"
              f" {e['black_weight']:>7.2f}")


def llm_context(args, st: Setup) -> dict:
    return {
        "capital": f"R${args.capital_brl:g} ({st.cash:.2f} USDT)",
        "max_lev": f"{st.limits.max_leverage:g}", "max_risk": f"{st.limits.max_risk_per_trade:g}",
        "daily_loss": f"{st.limits.daily_loss_limit:g}",
        "min_order": f"{st.spec.min_order_value(float(st.candles['close'].iloc[-1])):.2f}",
        "goal": "at least double the capital per day (+100%/day); ruin or liquidation kills the agent",
    }


if __name__ == "__main__":
    main()
