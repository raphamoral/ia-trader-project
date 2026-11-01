"""python -m trader.policy write   → a LLM local escreve a política agora (data/policy/current.json)
python -m trader.policy show    → mostra a política atual
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from trader.policy.context import build_context
from trader import llm
from trader.policy.writer import load_current, policy_to_dict, provider_chat, save, write_policy

SYMBOLS = ["SOL/USDT", "ETH/USDT", "XRP/USDT", "BTC/USDT"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "show"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--provider", choices=llm.PROVIDERS, help="ollama (padrão) ou anthropic (por token)")
    ap.add_argument("--model", help="modelo do provedor")
    args = ap.parse_args()
    data = Path(args.data)
    folder = data / "policy"
    current = load_current(folder, SYMBOLS)
    if args.cmd == "show":
        print(json.dumps(policy_to_dict(current), ensure_ascii=False, indent=2) if current else "sem política")
        return
    cfg = llm.config(args.provider, args.model)
    ctx = build_context(data, SYMBOLS)
    from trader.policy.schema import json_schema

    constrained = cfg.provider == "ollama"
    policy, report = write_policy(ctx, SYMBOLS, current, chat=provider_chat(cfg, json_schema(SYMBOLS) if constrained else None),
                                  lenient=constrained)
    report["provider"], report["model"] = cfg.provider, cfg.model
    report["usage"] = {"calls": llm.usage.calls, "input_tokens": llm.usage.input_tokens,
                       "output_tokens": llm.usage.output_tokens, "cost_usd": round(llm.usage.cost_usd, 4)}
    save(policy, report, folder, str(pd.Timestamp.now(tz="UTC")))
    print(json.dumps({"report": report, "policy": policy_to_dict(policy)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
