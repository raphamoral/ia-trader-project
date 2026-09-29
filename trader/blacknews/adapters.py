"""Ponte com o agente de notícias confiáveis (`trader.news`), só leitura.

Nada aqui escreve em data/news/ nem altera o módulo confiável. Se o módulo não
estiver presente, as funções devolvem vazio e o relatório diz isso.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

TRUSTED_COLUMNS = ["id", "source", "tier", "visible_at", "text_en", "assets"]


def trusted_module(name: str):
    """Importa trader.news.<name> se existir; None caso contrário."""
    try:
        return importlib.import_module(f"trader.news.{name}")
    except ImportError:
        return None


def _tier_number(value) -> int:
    s = str(value).lower().replace("tier", "").strip()
    return int(s) if s.isdigit() else 2


def load_trusted(
    start: pd.Timestamp | None = None, end: pd.Timestamp | None = None, data: Path | str = Path("data")
) -> pd.DataFrame:
    """Itens confiáveis normalizados: id, source, tier (1/2), visible_at, text_en, assets.

    Usa `trader.news.pipeline.load_items` (só rótulos em cache, nunca chama a LLM
    nem escreve nada) com a visibilidade oficial do módulo confiável.
    """
    pipeline, enrich = trusted_module("pipeline"), trusted_module("enrich")
    if pipeline is None or enrich is None:
        return pd.DataFrame(columns=TRUSTED_COLUMNS)
    df = pipeline.load_items(Path(data), enrich.Enricher(cache_dir=Path(data)))
    if df.empty:
        return pd.DataFrame(columns=TRUSTED_COLUMNS)
    # mesma convenção das fontes black: item coletado >30 min depois de publicado é histórico
    # (visível em published_at + latência), senão a primeira coleta do agente confiável
    # parece "atrasada" e toda fonte black vira falsa líder
    from trader.news.store import visible_at

    df = df.copy()
    df["backfilled"] = df["backfilled"].astype(bool) | ((df["first_seen_at"] - df["published_at"]) > pd.Timedelta(minutes=30))
    df["visible_at"] = visible_at(df)
    out = pd.DataFrame(
        {
            "id": df["id"].astype(str),
            "source": df["domain"],
            "tier": df["tier"].map(_tier_number),
            "visible_at": pd.to_datetime(df["visible_at"], utc=True),
            "text_en": df["translated_text"].where(df["translated_text"].notna(), df["original_text"]).astype(str),
            "assets": [a if a else q for a, q in zip(df["assets"].fillna(""), df["query_assets"].fillna(""))],
        }
    )
    if start is not None:
        out = out[out["visible_at"] >= start]
    if end is not None:
        out = out[out["visible_at"] < end]
    return out.sort_values("visible_at").reset_index(drop=True)


def trusted_feature_columns(columns: list[str]) -> list[str]:
    """Colunas do conjunto confiável: tudo que não é black_ (e não é chave)."""
    return [c for c in columns if not c.startswith("black_") and c not in ("ts", "asset")]


def black_feature_columns(columns: list[str], weighted: bool | None = None) -> list[str]:
    """weighted=None: todas as black_; False: só as não ponderadas; True: só black_rw_."""
    cols = [c for c in columns if c.startswith("black_")]
    if weighted is True:
        return [c for c in cols if c.startswith("black_rw_")]
    if weighted is False:
        return [c for c in cols if not c.startswith("black_rw_")]
    return cols
