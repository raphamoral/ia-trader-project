"""Reputação das fontes black: regras de crédito daqui, modelo do `trader.news.reputation`.

O modelo (Beta por fonte e tipo, prior BLACK_TIER começando baixo, decaimento,
point-in-time com resolved_at < t, piso de exploração, trava anti-manipulação)
é o do agente confiável, importado sem alteração. Aqui só se decide QUAIS
desfechos cada item black gera:

- lead confirmado por nível 1  → confirmation ✓ (peso 1) e timeliness ✓
  (peso 1 + bônus de antecedência até +0,5). É o sinal positivo principal.
- lead confirmado só por nível 2 → confirmation ✓ e timeliness ✓, peso 0,5.
- eco (chegou depois da confiável) → confirmation ✓ com peso 0,1 e
  timeliness ✗: repetir notícia pública não é informação, então crédito de
  pontualidade vai só para leads.
- afirmação sem confirmação no horizonte → confirmation ✗ no fim do horizonte.
Guardas locais: um crédito por (fonte, história); spam pesa metade; cópia de
texto de outra fonte não gera crédito.
Direção de preço fica fora de propósito: para as fontes black o sinal principal
é "boato que o nível 1 confirmou depois".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader.news.reputation import BLACK_TIER, OUTCOME_COLUMNS, Reputation, ReputationConfig


@dataclass(frozen=True)
class BlackCreditRules:
    horizon: pd.Timedelta = pd.Timedelta(hours=48)
    lead_bonus_cap_hours: float = 24.0
    echo_confirmation_weight: float = 0.1
    tier2_weight: float = 0.5
    spam_weight: float = 0.5
    promote_score: float = 0.6
    promote_min_leads: int = 5
    promote_min_days: int = 3


def black_outcomes(lags: pd.DataFrame, spam: pd.DataFrame | None = None, rules: BlackCreditRules = BlackCreditRules()) -> pd.DataFrame:
    """Desfechos no esquema OUTCOME_COLUMNS do trader.news.reputation."""
    df = lags.copy()
    if spam is not None and not spam.empty:
        df = df.merge(spam[["id", "is_spam", "dup_count"]], on="id", how="left")
    for col, default in (("is_spam", False), ("dup_count", 0)):
        if col not in df:
            df[col] = default
    rows = []
    credited: set[tuple[str, str]] = set()
    for _, r in df.sort_values("visible_at").iterrows():
        base = rules.spam_weight if bool(r["is_spam"]) else 1.0
        copied = (r["dup_count"] or 0) > 0
        story = str(r.get("story_id"))
        if r["status"] in ("lead", "echo"):
            if copied or (r["source"], story) in credited:
                continue
            credited.add((r["source"], story))
        if r["status"] == "lead":
            tier1 = pd.notna(r["first_tier1_at"])
            confirm_at = r["first_tier1_at"] if tier1 else r["first_trusted_at"]
            w = base * (1.0 if tier1 else rules.tier2_weight)
            lead_h = (confirm_at - r["visible_at"]).total_seconds() / 3600
            bonus = 0.5 * min(max(lead_h, 0.0) / rules.lead_bonus_cap_hours, 1.0) if tier1 else 0.0
            rows.append((r["source"], "confirmation", True, confirm_at, r["id"], w))
            rows.append((r["source"], "timeliness", True, confirm_at, r["id"], w * (1 + bonus)))
        elif r["status"] == "echo":
            rows.append((r["source"], "confirmation", True, r["visible_at"], r["id"], base * rules.echo_confirmation_weight))
            rows.append((r["source"], "timeliness", False, r["visible_at"], r["id"], base))
        elif r["status"] == "unconfirmed" and bool(r["is_claim"]):
            rows.append((r["source"], "confirmation", False, r["visible_at"] + rules.horizon, r["id"], base))
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def black_reputation(
    outcomes: pd.DataFrame, items: pd.DataFrame, config: ReputationConfig = ReputationConfig(), tier: int | None = BLACK_TIER
) -> Reputation:
    """Reputation do trader.news com as fontes no `tier` dado (black: BLACK_TIER, priori baixa;
    monitor de segurança: None, priori média). items alimenta a trava anti-manipulação."""
    tiers = {s: tier for s in items["source"].unique()}
    if "sentiment" not in items:
        items = items.assign(sentiment=-1.0)  # incidentes são sempre notícia negativa
    guard = items.rename(columns={"source": "domain"})[["domain", "visible_at", "sentiment"]].copy()
    guard["sentiment"] = pd.to_numeric(guard["sentiment"]).fillna(0.0)
    return Reputation(outcomes, tiers, config, items=guard)


def promotion_candidates(
    rep: Reputation, outcomes: pd.DataFrame, t: pd.Timestamp, rules: BlackCreditRules = BlackCreditRules()
) -> pd.DataFrame:
    """Só relatório. Nenhuma fonte black é movida para o conjunto confiável
    automaticamente: é a separação que mantém válido o experimento A/B/B'/C."""
    past = outcomes[outcomes["resolved_at"] < t]
    leads = past[(past["kind"] == "timeliness") & past["success"]]
    echoes = past[(past["kind"] == "timeliness") & ~past["success"]]
    sources = sorted(set(past["source"]))
    if not sources:
        return pd.DataFrame(columns=["source", "score", "leads", "lead_days", "echo_share"])
    scores = rep.scores(t, sources)
    rows = []
    for s in sources:
        n_lead = int((leads["source"] == s).sum())
        n_echo = int((echoes["source"] == s).sum())
        days = leads.loc[leads["source"] == s, "resolved_at"].dt.floor("D").nunique()
        echo_share = n_echo / (n_lead + n_echo) if n_lead + n_echo else 0.0
        rows.append({"source": s, "score": float(scores.loc[s, "score"]), "leads": n_lead, "lead_days": int(days),
                     "echo_share": echo_share})
    table = pd.DataFrame(rows)
    ok = (
        (table["score"] >= rules.promote_score)
        & (table["leads"] >= rules.promote_min_leads)
        & (table["lead_days"] >= rules.promote_min_days)
        & (table["echo_share"] < 0.5)
    )
    return table[ok].sort_values("score", ascending=False).reset_index(drop=True)
