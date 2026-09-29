"""Treino da LLM local.

    python -m trader.training creator-data
    python -m trader.training eval-creator --creators rules,llm:qwen2.5:7b --limit 60
    python -m trader.training signal-check --symbols BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT --horizon 6
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from trader.training import creator_data, signal_check
from trader.training.eval_creator import evaluate_creators


def main(argv: list[str] | None = None) -> None:
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="trader.training")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=8)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("creator-data", help="monta a base de treino do Criador a partir dos registros")
    build.add_argument("--horizon", type=int, default=3, help="janelas avaliadas após cada morte")
    build.add_argument("--out", type=Path, default=Path("data/training/creator"))
    build.add_argument(
        "--since", default="20260928-235135", help="ignora rodadas anteriores (antes da correção do rebalanceamento)"
    )

    ev = sub.add_parser("eval-creator", help="compara Criadores nas autópsias da validação")
    ev.add_argument("--creators", default="rules,llm:qwen2.5:7b")
    ev.add_argument("--limit", type=int, default=60)
    ev.add_argument("--dataset", type=Path, default=Path("data/training/creator"))
    ev.add_argument("--llm-url", default="http://localhost:11434/v1")

    sig = sub.add_parser("signal-check", help="as correções dirigidas batem mutação aleatória?")
    sig.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT")
    sig.add_argument("--horizon", type=int, default=6)
    sig.add_argument("--since", default="20260928-235135")
    sig.add_argument("--split", default="val", choices=["val", "train", "all"])
    sig.add_argument("--limit", type=int, default=300)

    args = parser.parse_args(argv)
    with ProcessPoolExecutor(args.workers, initializer=creator_data.init_worker, initargs=(str(args.data),)) as pool:
        creator_data.init_worker(str(args.data))
        if args.command == "creator-data":
            build_creator_data(args, pool)
        elif args.command == "signal-check":
            run_signal_check(args, pool)
        else:
            records = [json.loads(line) for line in open(args.dataset / "eval.jsonl", encoding="utf-8")][: args.limit]
            print(f"{len(records)} autópsias de validação")
            for report in evaluate_creators(records, args.creators.split(","), pool, args.llm_url):
                print(report.row())


def _swing_registries(data: Path, since: str) -> list[Path]:
    return [
        db for db in sorted((data / "evolution").glob("*.sqlite"))
        if db.stem >= since and "-day" not in db.stem  # day trade tem outro genoma
    ]


def run_signal_check(args: argparse.Namespace, pool) -> None:
    examples = [ex for db in _swing_registries(args.data, args.since) for ex in creator_data.load_examples(db, args.horizon)]
    if args.split != "all":
        examples = [ex for ex in examples if ex.split == args.split]
    examples = examples[: args.limit]
    symbols = args.symbols.split(",")
    print(f"{len(examples)} autópsias, horizonte {args.horizon} janelas, ativos: {', '.join(symbols)}")
    scores = [s for s in pool.map(signal_check.cross_asset_scores, [(ex, symbols) for ex in examples], chunksize=4) if s]
    v = signal_check.verdict(scores)
    for name, mean in sorted(v.means.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<10} nota média {mean:+.3f}")
    print(f"regras > mutação aleatória em {v.rules_vs_random:.0%} de {v.n} autópsias")
    print(f"diferença média regras − mutação: IC95% [{v.ci_low:+.3f}, {v.ci_high:+.3f}]")
    print("HÁ SINAL para treinar o Criador" if v.has_signal else "SEM SINAL: treinar agora ensinaria ruído")


def build_creator_data(args: argparse.Namespace, pool) -> None:
    dbs = _swing_registries(args.data, args.since)
    examples = [ex for db in dbs for ex in creator_data.load_examples(db, args.horizon)]
    print(f"{len(examples)} mortes com sucessor em {len(dbs)} rodadas")
    scores = creator_data.score_examples(examples, pool)
    records = creator_data.build_records(examples, scores)
    for name, rows in records.items():
        creator_data.write_jsonl(args.out / f"{name}.jsonl", rows)
        print(f"  {name}: {len(rows)}")

    good = Counter(r["meta"]["source"] for r in records["sft_train"] + records["sft_val"])
    real_good = sum(creator_data.is_good(s, "successor") for s in scores)
    print(f"sucessores reais que superaram pai e referência: {real_good}/{len(scores)} ({real_good / max(len(scores), 1):.0%})")
    print("origem da melhor resposta nos exemplos bons:", dict(good))
    by_cause = Counter(ex.autopsy["cause"] for ex in examples)
    print("mortes por causa:", dict(by_cause))


if __name__ == "__main__":
    main()
