"""Experimento em paralelo: as fontes black melhoram ou pioram o resultado?

Braços, todos com as MESMAS sementes, dados e janelas:
  A  = só features confiáveis
  B  = confiáveis + black_ sem ponderação
  B' = confiáveis + black_rw_ (ponderadas pela reputação da fonte)
  C  = só black_ (opcional)

`evaluate(arm, feature_columns, seed)` é a evolução real (ou um stub nos
testes) e devolve um dict com:
  score (nota do holdout: retorno / queda máxima), goal_hit_rate, ruin,
  e opcionalmente pnl: pd.Series de PnL por instante do holdout.

Comparações pareadas por semente: diferença média, IC 95% por bootstrap e
teste do sinal exato. O veredito só sai quando há evidência; com poucas
sementes, diz que não há.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Callable

import numpy as np
import pandas as pd

from trader.blacknews.adapters import black_feature_columns, trusted_feature_columns

Evaluate = Callable[[str, list[str], int], dict]
METRICS = ("score", "goal_hit_rate", "ruin")


def arms_for(columns: list[str], include_c: bool = True) -> dict[str, list[str]]:
    trusted = trusted_feature_columns(columns)
    arms = {
        "A": trusted,
        "B": trusted + black_feature_columns(columns, weighted=False),
        "B'": trusted + black_feature_columns(columns, weighted=True),
    }
    if include_c:
        arms["C"] = black_feature_columns(columns)
    return arms


def sign_test_p(diffs: np.ndarray) -> float:
    """Teste do sinal exato bilateral (empates descartados)."""
    d = diffs[diffs != 0]
    n, k = len(d), int((d > 0).sum())
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    if len(diffs) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


@dataclass
class Comparison:
    treatment: str
    control: str
    metric: str
    n: int
    mean_diff: float
    ci: tuple[float, float]
    p_sign: float
    verdict: str


@dataclass
class ExperimentReport:
    results: pd.DataFrame  # uma linha por (arm, seed)
    summary: pd.DataFrame  # média, desvio, mediana por braço e métrica
    comparisons: list[Comparison]
    leakage: dict[str, dict] = field(default_factory=dict)

    def text(self) -> str:
        lines = ["Resumo por braço (média ± desvio, mediana):"]
        for arm, g in self.summary.groupby("arm", sort=False):
            parts = [f"{r.metric}={r.mean:+.3f}±{r.std:.3f} (med {r.median:+.3f})" for r in g.itertuples()]
            lines.append(f"  {arm:3s} " + "  ".join(parts))
        lines.append("Pareado por semente (tratamento − controle):")
        for c in self.comparisons:
            lines.append(
                f"  {c.treatment} vs {c.control} [{c.metric}] n={c.n} Δ={c.mean_diff:+.3f} "
                f"IC95%=[{c.ci[0]:+.3f}, {c.ci[1]:+.3f}] p(sinal)={c.p_sign:.3f} → {c.verdict}"
            )
        for arm, info in self.leakage.items():
            flag = "SUSPEITO" if info["flag"] else "ok"
            lines.append(
                f"  vazamento {arm}: {info['gain_share']:.0%} do ganho sobre A ocorre em {info['time_share']:.0%} "
                f"do tempo perto de carimbos suspeitos → {flag}"
            )
        return "\n".join(lines)


def _verdict(n: int, ci: tuple[float, float], p: float, higher_is_better: bool = True) -> str:
    if n < 6:
        return f"evidência insuficiente (n={n}; o teste do sinal precisa de ≥6 sementes para p<0,05)"
    better = ci[0] > 0 if higher_is_better else ci[1] < 0
    worse = ci[1] < 0 if higher_is_better else ci[0] > 0
    if better and p < 0.05:
        return "melhora"
    if worse and p < 0.05:
        return "piora"
    return "evidência insuficiente (IC cruza zero ou p≥0,05)"


def leakage_smell(
    pnl_treatment: pd.Series, pnl_control: pd.Series, suspicious: pd.DatetimeIndex, window: pd.Timedelta
) -> dict:
    """Quanto do ganho extra do tratamento cai perto de carimbos suspeitos.

    Se uma fatia pequena do tempo concentra a maior parte do ganho, o braço
    pode estar lucrando com timestamp errado (vazamento), não com informação.
    """
    diff = pnl_treatment.sub(pnl_control, fill_value=0).sort_index()
    gain = diff.clip(lower=0)
    if gain.sum() <= 0 or len(suspicious) == 0:
        return {"gain_share": 0.0, "time_share": 0.0, "flag": False}
    idx = diff.index.tz_convert("UTC").tz_localize(None).to_numpy() if diff.index.tz is not None else diff.index.to_numpy()
    near = np.zeros(len(idx), dtype=bool)
    w = np.timedelta64(int(window.total_seconds()), "s")
    sus = suspicious.tz_convert("UTC").tz_localize(None) if suspicious.tz is not None else suspicious
    for s in sus.to_numpy():
        lo, hi = np.searchsorted(idx, s - w), np.searchsorted(idx, s + w, "right")
        near[lo:hi] = True
    gain_share = float(gain[near].sum() / gain.sum())
    time_share = float(near.mean())
    return {"gain_share": gain_share, "time_share": time_share, "flag": gain_share > 0.3 and gain_share > 3 * time_share}


DAYTRADE_PAIRS = [("W", "A"), ("WB", "W"), ("WB'", "W"), ("WB'", "WB")]


def daytrade_arms(white: list[str], black_unweighted: list[str], black_weighted: list[str]) -> dict[str, list[str]]:
    """Braços do day trade: A sem notícias, W só confiáveis, WB + black, WB' + black ponderada."""
    bad = [c for c in white if c.startswith("black_")]
    if bad:
        raise ValueError(f"coluna black no conjunto confiável: {bad}")
    return {"A": [], "W": list(white), "WB": [*white, *black_unweighted], "WB'": [*white, *black_weighted]}


def run_experiment(
    evaluate: Evaluate,
    feature_columns: list[str],
    seeds: list[int],
    include_c: bool = True,
    suspicious: pd.DatetimeIndex | None = None,
    suspicious_window: pd.Timedelta = pd.Timedelta(hours=2),
    arms: dict[str, list[str]] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    leak_control: str = "A",
) -> ExperimentReport:
    """arms/pairs opcionais: por padrão A/B/B'/C a partir de `feature_columns`;
    para o day trade use arms=daytrade_arms(...), pairs=DAYTRADE_PAIRS, leak_control="W"."""
    arms = arms or arms_for(feature_columns, include_c)
    rows, pnls = [], {}
    for seed in seeds:  # mesma semente em todos os braços
        for arm, cols in arms.items():
            out = evaluate(arm, cols, seed)
            rows.append({"arm": arm, "seed": seed, **{m: float(out.get(m, np.nan)) for m in METRICS}})
            if isinstance(out.get("pnl"), pd.Series):
                pnls[(arm, seed)] = out["pnl"]
    results = pd.DataFrame(rows)
    summary = (
        results.melt(id_vars=["arm", "seed"], value_vars=list(METRICS), var_name="metric")
        .groupby(["arm", "metric"], sort=False)["value"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    comparisons = []
    if pairs is None:
        pairs = [("B", "A"), ("B'", "A"), ("B'", "B")] + ([("C", "A")] if "C" in arms else [])
    for treat, ctrl in pairs:
        for metric in METRICS:
            t = results[results["arm"] == treat].set_index("seed")[metric]
            c = results[results["arm"] == ctrl].set_index("seed")[metric]
            d = (t - c).dropna().to_numpy()
            ci = bootstrap_ci(d)
            p = sign_test_p(d)
            comparisons.append(
                Comparison(treat, ctrl, metric, len(d), float(d.mean()) if len(d) else np.nan, ci, p,
                           _verdict(len(d), ci, p, higher_is_better=metric != "ruin"))
            )
    leakage = {}
    if suspicious is not None and pnls:
        for arm in [a for a in arms if a != leak_control and any(c.startswith("black_") for c in arms[a])]:
            shares = [
                leakage_smell(pnls[(arm, s)], pnls[(leak_control, s)], suspicious, suspicious_window)
                for s in seeds
                if (arm, s) in pnls and (leak_control, s) in pnls
            ]
            if shares:
                leakage[arm] = {
                    "gain_share": float(np.mean([x["gain_share"] for x in shares])),
                    "time_share": float(np.mean([x["time_share"] for x in shares])),
                    "flag": sum(x["flag"] for x in shares) > len(shares) / 2,
                }
    return ExperimentReport(results, summary, comparisons, leakage)


def suspicious_timestamps(items: pd.DataFrame) -> pd.DatetimeIndex:
    """Carimbos em que não dá para confiar: publicado depois de coletado (relógio
    errado), hora cheia exata vinda do autor (provável data sem hora), ou blog/site
    coletado muito depois de publicado (texto pode ter sido editado depois; posts
    do 4chan não são editáveis)."""
    pub, seen = items["published_at"], items["first_seen_at"]
    odd = (pub > seen) | ((items["ts_reliability"] == "author") & (pub.dt.floor("h") == pub))
    late = ((seen - pub) > pd.Timedelta(days=1)) & items["source_kind"].isin(["blog", "farm"])
    return pd.DatetimeIndex(items.loc[odd | late, "visible_at"]).sort_values()


DEMO_COLUMNS = ["news_sent_4h", "news_count_4h", "black_count", "black_mean_sent", "black_rw_count", "black_rw_mean_sent"]


def stub_evaluate(arm: str, columns: list[str], seed: int) -> dict:
    """Avaliação sintética para demonstrar o harness (não é a evolução real).

    A mesma semente dá o mesmo "mercado" a todos os braços; black sem ponderação
    só acrescenta ruído, a ponderada acrescenta um ganho pequeno.
    """
    market = np.random.default_rng(seed)
    base = 1.0 + market.normal(0, 0.4)
    noise = np.random.default_rng([seed, sum(map(ord, arm))])
    has_trusted = any(not c.startswith("black_") for c in columns)
    score = base if has_trusted else base - 0.6
    if any(c.startswith("black_") and not c.startswith("black_rw_") for c in columns):
        score += noise.normal(0, 0.25)
    if any(c.startswith("black_rw_") for c in columns):
        score += 0.15 + noise.normal(0, 0.1)
    return {"score": score, "goal_hit_rate": 0.0, "ruin": float(noise.random() < 0.05)}
