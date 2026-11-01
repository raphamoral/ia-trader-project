"""Reputação de fontes aprendida com resultados, estritamente point-in-time.

API (usada também pelo agente de notícias "black"):

    outcomes = direction_outcomes(items, prices, horizon)     # acertou a direção?
    outcomes += story_outcomes(items)                          # confirmada por nível 1? chegou antes?
    rep = Reputation(outcomes, tiers={"reuters.com": 1, "t.me/canal": BLACK_TIER})
    table = rep.as_of(t)                                       # posterior por fonte e tipo, só com o passado
    w = rep.weights(t, ["reuters.com", "t.me/canal"])          # peso para agregar features
    chosen = rep.priority(t, candidates, budget=50, rng=rng)   # quem ganha orçamento de LLM

Modelo: Beta(α, β) por fonte e por tipo de resultado, com prior dado pelo
nível da fonte, decaimento exponencial (meia-vida) para a reputação poder
cair, e encolhimento para o prior enquanto houver poucas amostras.

Point-in-time: um resultado só conta depois de `resolved_at` (o fim do
horizonte, ou o momento da confirmação). `as_of(t)` usa apenas resultados com
resolved_at < t.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

BLACK_TIER = 3  # fontes não verificadas (rumores, canais, redes)
KINDS = ("direction", "confirmation", "timeliness")
KIND_WEIGHT = {"direction": 0.5, "confirmation": 0.35, "timeliness": 0.15}
PRIORS: dict[int | None, tuple[float, float]] = {1: (8.0, 2.0), 2: (4.0, 2.5), BLACK_TIER: (1.0, 4.0), None: (1.0, 1.5)}
OUTCOME_COLUMNS = ["source", "kind", "success", "resolved_at", "item_id", "weight"]


@dataclass(frozen=True)
class ReputationConfig:
    half_life: pd.Timedelta = pd.Timedelta(days=90)
    min_samples: float = 10.0  # abaixo disso a média fica perto do prior
    explore_floor: float = 0.35  # peso mínimo de fontes novas/pouco testadas
    max_share: float = 0.25  # nenhuma fonte passa desta fatia do peso total numa agregação
    burst_window: pd.Timedelta = pd.Timedelta(hours=6)
    burst_min_items: int = 5
    burst_one_sided: float = 0.9  # fração de itens com o mesmo sinal para contar como rajada
    burst_flag_for: pd.Timedelta = pd.Timedelta(days=7)
    flagged_decay_mult: float = 4.0  # fonte sinalizada esquece o passado bom 4x mais rápido
    flagged_weight_mult: float = 0.5


class Reputation:
    def __init__(
        self,
        outcomes: pd.DataFrame,
        tiers: dict[str, int | None],
        config: ReputationConfig = ReputationConfig(),
        items: pd.DataFrame | None = None,
    ) -> None:
        """items (opcional): notícias com source/domain, visible_at, sentiment; usadas pela trava anti-manipulação."""
        self.outcomes = outcomes.sort_values("resolved_at").reset_index(drop=True) if len(outcomes) else outcomes
        self.tiers = tiers
        self.config = config
        self.items = items

    def prior(self, source: str) -> tuple[float, float]:
        return PRIORS.get(self.tiers.get(source), PRIORS[None])

    def flagged(self, t: pd.Timestamp) -> set[str]:
        """Fontes com rajada recente de alegações do mesmo lado (só itens visíveis antes de t)."""
        if self.items is None or self.items.empty:
            return set()
        c = self.config
        recent = self.items[(self.items["visible_at"] < t) & (self.items["visible_at"] >= t - c.burst_flag_for)]
        out = set()
        for source, grp in recent.groupby("domain"):
            grp = grp.sort_values("visible_at")
            times = grp["visible_at"].to_numpy()
            signs = np.sign(grp["sentiment"].to_numpy())
            lo = 0
            for hi in range(len(times)):
                while times[hi] - times[lo] > c.burst_window.to_timedelta64():
                    lo += 1
                window = signs[lo : hi + 1]
                window = window[window != 0]
                if len(window) >= c.burst_min_items and max((window > 0).mean(), (window < 0).mean()) >= c.burst_one_sided:
                    out.add(source)
                    break
        return out

    def as_of(self, t: pd.Timestamp, sources: list[str] | None = None) -> pd.DataFrame:
        """Posterior por (fonte, tipo) com resultados resolvidos antes de t."""
        c = self.config
        flagged = self.flagged(t)
        past = self.outcomes[self.outcomes["resolved_at"] < t] if len(self.outcomes) else self.outcomes
        names = sorted(set(sources or []) | set(past["source"] if len(past) else []))
        rows = []
        for source in names:
            a0, b0 = self.prior(source)
            half = c.half_life / (c.flagged_decay_mult if source in flagged else 1.0)
            mine = past[past["source"] == source] if len(past) else past
            for kind in KINDS:
                sub = mine[mine["kind"] == kind] if len(mine) else mine
                if len(sub):
                    age = (t - sub["resolved_at"]) / half
                    w = sub["weight"].to_numpy() * np.power(0.5, age.to_numpy(dtype=float))
                    wins = float((w * sub["success"].to_numpy()).sum())
                    n = float(w.sum())
                else:
                    wins = n = 0.0
                alpha, beta = a0 + wins, b0 + (n - wins)
                prior_mean, post_mean = a0 / (a0 + b0), alpha / (alpha + beta)
                shrink = min(1.0, n / c.min_samples)
                rows.append(
                    {"source": source, "kind": kind, "alpha": alpha, "beta": beta, "n_eff": n,
                     "mean": prior_mean + (post_mean - prior_mean) * shrink, "flagged": source in flagged}
                )
        return pd.DataFrame(rows, columns=["source", "kind", "alpha", "beta", "n_eff", "mean", "flagged"])

    def scores(self, t: pd.Timestamp, sources: list[str]) -> pd.DataFrame:
        """Nota combinada (0..1) por fonte, com n_eff total e sinalização."""
        table = self.as_of(t, sources)
        table["w"] = table["kind"].map(KIND_WEIGHT)
        grouped = table.groupby("source").apply(
            lambda g: pd.Series({"score": float((g["mean"] * g["w"]).sum() / g["w"].sum()),
                                 "n_eff": float(g["n_eff"].sum()), "flagged": bool(g["flagged"].any())}),
            include_groups=False,
        )
        return grouped.reindex(sources)

    def weights(self, t: pd.Timestamp, sources: list[str]) -> pd.Series:
        """Peso determinístico para agregação: nota, com piso otimista para fontes pouco testadas."""
        s = self.scores(t, sorted(set(sources)))
        c = self.config
        w = s["score"].copy()
        new = s["n_eff"] < c.min_samples
        w[new] = np.maximum(w[new], c.explore_floor)
        w[s["flagged"]] *= c.flagged_weight_mult
        return w.reindex(sources).fillna(c.explore_floor)

    def priority(
        self, t: pd.Timestamp, candidates: pd.DataFrame, budget: int, rng: random.Random, explore_share: float = 0.2
    ) -> pd.Index:
        """Escolhe quais itens recebem orçamento de LLM: maioria por reputação, parte por amostragem de Thompson."""
        if len(candidates) <= budget:
            return candidates.index
        sources = candidates["domain"].tolist()
        s = self.scores(t, sorted(set(sources)))
        table = self.as_of(t, sorted(set(sources)))
        exploit_n = int(round(budget * (1 - explore_share)))
        ranked = candidates.assign(_score=candidates["domain"].map(s["score"]))
        chosen = list(ranked.sort_values("_score", ascending=False).index[:exploit_n])
        rest = ranked.drop(index=chosen)
        # Thompson: amostra uma nota da Beta de cada fonte; fontes pouco vistas às vezes ganham
        post = table.groupby("source")[["alpha", "beta"]].sum()
        draws = {src: rng.betavariate(post.at[src, "alpha"], post.at[src, "beta"]) for src in post.index}
        rest = rest.assign(_draw=rest["domain"].map(draws))
        chosen += list(rest.sort_values("_draw", ascending=False).index[: budget - exploit_n])
        return pd.Index(chosen)

    def history(self, dates: list[pd.Timestamp]) -> pd.DataFrame:
        """Série temporal da reputação (para persistir em Parquet)."""
        frames = [self.as_of(d).assign(as_of=d) for d in dates]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cap_shares(weights: pd.Series, max_share: float) -> pd.Series:
    """Limita a fatia de cada fonte no peso total, redistribuindo o excesso entre as demais.

    Mantém a soma original. Se o limite for impossível (max_share * n < 1), iguala todas.
    """
    w = weights.astype(float).clip(lower=0)
    total = w.sum()
    if total == 0:
        return w
    if max_share * len(w) < 1:
        return pd.Series(total / len(w), index=w.index)
    share = w / total
    capped = pd.Series(False, index=w.index)
    while True:
        over = (share > max_share + 1e-12) & ~capped
        if not over.any():
            break
        capped |= over
        share[capped] = max_share
        free = ~capped
        remaining = 1 - max_share * capped.sum()
        share[free] = share[free] / share[free].sum() * remaining if share[free].sum() > 0 else remaining / free.sum()
    return share * total


def direction_outcomes(
    items: pd.DataFrame,
    prices: dict[str, pd.Series],
    horizon: pd.Timedelta = pd.Timedelta(hours=4),
    noise: float = 0.002,
    min_abs_sentiment: float = 0.2,
) -> pd.DataFrame:
    """Acertou a direção? Resolve em visible_at + horizon. Movimentos abaixo de `noise` não contam.

    prices: ativo -> série de preços de fechamento indexada pelo horário em que o preço é conhecido.
    """
    strong = items[items["sentiment"].abs() >= min_abs_sentiment]
    if strong.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    pairs = strong.assign(asset=strong["assets"].fillna("").str.split(",")).explode("asset")
    frames = []
    for asset, grp in pairs.groupby("asset"):
        p = prices.get(asset)
        if p is None or len(p) < 2:
            continue
        idx = pd.DatetimeIndex(p.index).tz_convert(None).to_numpy(dtype="datetime64[ns]")
        stale = 2 * np.median(np.diff(idx))
        vals = p.to_numpy(float)
        t0 = grp["visible_at"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
        t1 = t0 + np.timedelta64(int(horizon.total_seconds() * 1e9), "ns")
        i0 = np.searchsorted(idx, t0, "right") - 1
        i1 = np.searchsorted(idx, t1, "right") - 1
        ok = (i0 >= 0) & (i1 > i0) & (idx[np.clip(i1, 0, None)] >= t1 - stale)
        move = np.where(ok, vals[np.clip(i1, 0, None)] / vals[np.clip(i0, 0, None)] - 1, 0.0)
        ok &= np.abs(move) >= noise
        g = grp[ok]
        frames.append(pd.DataFrame({
            "source": g["domain"].to_numpy(), "kind": "direction",
            "success": np.sign(move[ok]) == np.sign(g["sentiment"].to_numpy()),
            "resolved_at": pd.to_datetime(t1[ok], utc=True), "item_id": g["id"].to_numpy(), "weight": 1.0,
        }))
    return pd.concat(frames, ignore_index=True)[OUTCOME_COLUMNS] if frames else pd.DataFrame(columns=OUTCOME_COLUMNS)


def story_outcomes(
    items: pd.DataFrame, confirm_window: pd.Timedelta = pd.Timedelta(hours=48), retraction_weight: float = 3.0
) -> pd.DataFrame:
    """Confirmação e pontualidade a partir das histórias (precisa de story_id e tier).

    - confirmation: item de nível >1 cuja história teve relato de nível 1 dentro da janela.
      Sucesso resolve no horário do relato de nível 1; falha resolve no fim da janela.
    - timeliness: em histórias confirmadas, o item chegou antes do primeiro relato de nível 1?
    - itens com retracted=True (se a coluna existir) contam como falha de confirmação com peso extra.
    """
    first_t1 = items[items["tier"] == 1].groupby("story_id")["visible_at"].min()
    it = items[(items["tier"] != 1) & (items["tier"] != 9)]
    if it.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    t1 = pd.Series(first_t1.reindex(it["story_id"].to_numpy()).to_numpy(), index=it.index)
    t1 = pd.to_datetime(t1, utc=True)
    vis = it["visible_at"]
    confirmed = t1.notna() & (t1 - vis <= confirm_window)
    retracted = it["retracted"].fillna(False).astype(bool) if "retracted" in it.columns else pd.Series(False, index=it.index)
    t1_safe = t1.where(confirmed, vis)
    resolved = t1_safe.where(t1_safe > vis, vis).where(confirmed, vis + confirm_window)
    conf = pd.DataFrame({
        "source": it["domain"], "kind": "confirmation", "success": confirmed & ~retracted,
        "resolved_at": resolved.where(~retracted, vis + confirm_window), "item_id": it["id"],
        "weight": np.where(retracted, retraction_weight, 1.0),
    })
    keep = confirmed & ~retracted
    timely = pd.DataFrame({
        "source": it["domain"][keep], "kind": "timeliness", "success": (vis < t1)[keep],
        "resolved_at": resolved[keep], "item_id": it["id"][keep], "weight": 1.0,
    })
    return pd.concat([conf, timely], ignore_index=True)[OUTCOME_COLUMNS]
