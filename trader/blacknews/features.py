"""Features black_ numa grade de candles, calculadas só com o que era visível em cada instante.

Prefixos:
- black_*    : sem ponderação (braço B do experimento)
- black_rw_* : ponderadas pela reputação da fonte naquele instante (braço B'),
               incluindo black_rw_lead_*: só fontes que, até aquele dia, se
               mostraram líderes (chegam antes do nível 1) e não ecos.
Nenhuma coluna daqui pode sair sem o prefixo black_ (teste de separação).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader.blacknews.leadlag import CRYPTO
from trader.blacknews.schema import utc64

CRYPTO_ASSETS = CRYPTO


COLUMNS = [
    "black_count", "black_count_z", "black_mean_sent", "black_bull_share", "black_bear_share",
    "black_spam_share", "black_dup_share", "black_lang_spread", "black_source_spread",
    "black_confirmed_rate", "black_lead_minutes_median",
    "black_rw_count", "black_rw_mean_sent", "black_rw_lead_count", "black_rw_lead_score",
]


def _matches_asset(assets: str, asset: str) -> bool:
    s = set(filter(None, (assets or "").split(",")))
    return asset in s or (asset in CRYPTO_ASSETS and "CRYPTO" in s)


def build_features(
    items: pd.DataFrame,
    grid: pd.DatetimeIndex,
    asset: str,
    lags: pd.DataFrame | None = None,
    reputation=None,
    lookback: pd.Timedelta = pd.Timedelta(hours=4),
    baseline: pd.Timedelta = pd.Timedelta(days=7),
    outcomes: pd.DataFrame | None = None,
    leader_min_score: float = 0.205,  # acima da média da priori black (0,2): a reputação subiu
    leader_min_lead_share: float = 0.5,
) -> pd.DataFrame:
    """items: saída de spam_scores (enriquecidos). grid: instantes de decisão (as_of).
    reputation: `trader.news.reputation.Reputation` (ver reputation.black_reputation), opcional.
    outcomes: desfechos de reputation.black_outcomes, para decidir quem é líder
    (≥1 lead resolvido e leads/(leads+ecos) ≥ leader_min_lead_share, nota ≥ leader_min_score).
    A nota do trader.news encolhe para a priori com poucas amostras (min_samples=10), por isso o
    limiar é "acima da priori", não um valor alto; a ponderação usa a própria nota."""
    df = items[items["assets"].map(lambda a: _matches_asset(a, asset))].sort_values("visible_at").reset_index(drop=True)
    if lags is not None and not lags.empty:
        df = df.merge(lags[["id", "first_trusted_at", "status"]], on="id", how="left")
    else:
        df["first_trusted_at"], df["status"] = pd.NaT, None
    vis = utc64(df["visible_at"])
    sent = df["sentiment"].astype(float).fillna(0).to_numpy()
    spam = df["is_spam"].astype(bool).to_numpy() if "is_spam" in df else np.zeros(len(df), bool)
    dup = (df["dup_count"].to_numpy() > 0) if "dup_count" in df else np.zeros(len(df), bool)
    stance = df["stance"].fillna("neutral").to_numpy()
    claim = df["is_claim"].fillna(False).astype(bool).to_numpy()
    confirmed_at = utc64(df["first_trusted_at"])

    # reputação em fotografias diárias: a de cada dia usa só desfechos resolvidos antes da meia-noite
    rep_by_day: dict[pd.Timestamp, dict[str, float]] = {}
    all_sources = sorted(df["source"].unique())

    leaders_by_day: dict[pd.Timestamp, dict[str, float]] = {}
    timely = outcomes[outcomes["kind"] == "timeliness"] if outcomes is not None and len(outcomes) else None

    def weights_at(t: pd.Timestamp) -> dict[str, float]:
        day = t.floor("D")
        if day not in rep_by_day:
            rep_by_day[day] = (
                reputation.weights(day, all_sources).to_dict() if reputation is not None and all_sources else {}
            )
        return rep_by_day[day]

    def leaders_at(t: pd.Timestamp) -> dict[str, float]:
        """Fonte → nota, só para líderes conhecidos antes do dia de t."""
        day = t.floor("D")
        if day not in leaders_by_day:
            found: dict[str, float] = {}
            if timely is not None and reputation is not None:
                past = timely[timely["resolved_at"] < day]
                stats = past.groupby("source")["success"].agg(["sum", "count"])
                stats = stats[(stats["sum"] >= 1) & (stats["sum"] / stats["count"] >= leader_min_lead_share)]
                if len(stats):
                    scores = reputation.scores(day, list(stats.index))["score"]
                    found = {s: float(v) for s, v in scores.items() if v >= leader_min_score}
            leaders_by_day[day] = found
        return leaders_by_day[day]

    prior_weight = 0.2  # média da priori black para fonte sem histórico
    rows = []
    lb = np.timedelta64(int(lookback.total_seconds()), "s")
    base = np.timedelta64(int(baseline.total_seconds()), "s")
    n_buckets = max(int(baseline / lookback), 2)
    zero = dict.fromkeys(COLUMNS, 0.0)
    for t in grid:
        t64 = t.to_datetime64()
        hi = np.searchsorted(vis, t64, "right")  # visible_at <= t
        lo = np.searchsorted(vis, t64 - lb, "right")
        lo_base = np.searchsorted(vis, t64 - base, "right")
        if hi == lo_base:  # nada visível na base: atalho (a grade pode ter anos sem dado black)
            rows.append({"ts": t, **zero})
            continue
        sl = slice(lo, hi)
        n = hi - lo
        clean = ~spam[sl]
        # pico de volume: contagem da janela contra as janelas anteriores dentro da base
        past = (hi - lo_base - n) / (n_buckets - 1)
        edges = [np.searchsorted(vis, t64 - lb * (k + 1), "right") for k in range(n_buckets)]
        counts = np.array([edges[k - 1] - edges[k] if k else hi - edges[0] for k in range(1, n_buckets)])
        std = counts.std() if len(counts) > 1 else 0.0
        row = {
            "ts": t,
            "black_count": n,
            "black_count_z": (n - past) / std if std > 0 else 0.0,
            "black_mean_sent": float(sent[sl][clean].mean()) if clean.any() else 0.0,
            "black_bull_share": float((stance[sl] == "bullish").mean()) if n else 0.0,
            "black_bear_share": float((stance[sl] == "bearish").mean()) if n else 0.0,
            "black_spam_share": float(spam[sl].mean()) if n else 0.0,
            "black_dup_share": float(dup[sl].mean()) if n else 0.0,
            "black_lang_spread": int(df["original_lang"].iloc[sl].nunique()) if n else 0,
            "black_source_spread": int(df["source"].iloc[sl].nunique()) if n else 0,
        }
        # boatos confirmados: afirmações dos últimos 7 dias cuja confirmação já era visível em t
        bsl = slice(lo_base, hi)
        claims = claim[bsl]
        conf = claims & (confirmed_at[bsl] <= t64)
        row["black_confirmed_rate"] = float(conf.sum() / claims.sum()) if claims.any() else 0.0
        leads_min = (confirmed_at[bsl][conf] - vis[bsl][conf]) / np.timedelta64(1, "m")
        leads_min = leads_min[leads_min > 0]
        row["black_lead_minutes_median"] = float(np.median(leads_min)) if len(leads_min) else 0.0
        # ponderadas pela reputação do dia
        w_map = weights_at(t)
        w = np.array([w_map.get(s, prior_weight) for s in df["source"].iloc[sl]]) if n else np.array([])
        w = w * clean if n else w
        row["black_rw_count"] = float(w.sum()) if n else 0.0
        row["black_rw_mean_sent"] = float((w * sent[sl]).sum() / w.sum()) if n and w.sum() > 0 else 0.0
        # líderes: fontes que costumam chegar antes do nível 1 (ecos excluídos)
        lead_map = leaders_at(t)
        if n and lead_map:
            lw = np.array([lead_map.get(s, 0.0) for s in df["source"].iloc[sl]]) * clean
            row["black_rw_lead_count"] = float((lw > 0).sum())
            row["black_rw_lead_score"] = float((lw * sent[sl]).sum())
        else:
            row["black_rw_lead_count"] = row["black_rw_lead_score"] = 0.0
        rows.append(row)
    out = pd.DataFrame(rows, columns=["ts", *COLUMNS])
    assert all(c == "ts" or c.startswith("black_") for c in out.columns)
    return out
