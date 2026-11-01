"""Peça 4, parte 2: classificador rápido que imita os rótulos da Qwen (destilação), em < 1 ms.

Texto → TF-IDF (palavras e pedaços de palavras, funciona com slugs e várias línguas) →
regressões logísticas (relevante, tipo, direção do sentimento, moedas) e Ridge (gravidade).
Avaliação: treina nas manchetes até 2025, testa nas de 2026 (nunca vistas), e
compara com o reflexo por palavras-chave (news_reflex.classify) contra a Qwen.

  python -m trader.realtime.news_model train
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trader.realtime.news_labels import SYMBOLS, cache_path

MODEL_FILE = Path("models") / "news_reflex.pkl"
SENT_BANDS = (-0.2, 0.2)


def sent_class(x: float) -> int:
    return -1 if x <= SENT_BANDS[0] else 1 if x >= SENT_BANDS[1] else 0


class _Const:
    """Quando o treino só tem uma classe: prevê sempre ela."""

    def __init__(self, value) -> None:
        self.value, self.classes_ = value, np.array([value])

    def predict(self, X):
        return np.array([self.value] * X.shape[0])

    def predict_proba(self, X):
        return np.ones((X.shape[0], 1))


def _logit(X, y, **kw):
    from sklearn.linear_model import LogisticRegression

    y = pd.Series(y)
    return _Const(y.iloc[0]) if y.nunique() < 2 else LogisticRegression(**kw).fit(X, y)


class NewsModel:
    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self.vec = FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                  max_features=40000)),
        ])

    def fit(self, df: pd.DataFrame) -> "NewsModel":
        from sklearn.linear_model import Ridge

        X = self.vec.fit_transform(df["text"].str.lower())
        lr = dict(max_iter=2000, C=4.0, class_weight="balanced")
        self.relevant = _logit(X, df["relevant"].astype(int), **lr)
        rel = df["relevant"].to_numpy()
        Xr = X[rel]
        self.kind = _logit(Xr, df.loc[rel, "kind"], **lr)
        self.sent = _logit(Xr, df.loc[rel, "sentiment"].map(sent_class), **lr)
        self.sev = Ridge(alpha=1.0).fit(Xr, df.loc[rel, "severity"])
        self.syms = {}
        for s in (*SYMBOLS, "MARKET"):
            y = df.loc[rel, "symbols"].map(lambda v: s in v).astype(int)
            self.syms[s] = _logit(Xr, y, **lr) if 5 <= y.sum() < len(y) - 5 else None
        return self

    def predict(self, texts: list[str]) -> list[dict]:
        X = self.vec.transform([t.lower() for t in texts])
        rp = self.relevant.predict_proba(X)
        rel = rp[:, -1] if rp.shape[1] > 1 else np.full(len(texts), float(self.relevant.classes_[0]))
        kind = self.kind.predict(X)
        sp = self.sent.predict_proba(X)
        classes = list(self.sent.classes_)
        sent = sum(sp[:, classes.index(c)] * c for c in (1, -1) if c in classes) if sp.shape[1] > 1 \
            else np.full(len(texts), float(classes[0]))
        sev = np.clip(self.sev.predict(X), 0, 1)
        sym_p = {s: (m.predict_proba(X)[:, -1] if m is not None and len(m.classes_) > 1
                     else np.zeros(len(texts))) for s, m in self.syms.items()}
        out = []
        for i in range(len(texts)):
            syms = [s for s in SYMBOLS if sym_p[s][i] >= 0.5]
            out.append({"relevant": bool(rel[i] >= 0.5), "p_relevant": float(rel[i]), "kind": str(kind[i]),
                        "sentiment": float(sent[i]), "severity": float(sev[i]),
                        "symbols": syms or (["MARKET"] if sym_p["MARKET"][i] >= 0.5 else [])})
        return out


def load_labels(data: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in cache_path(data).read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows).drop_duplicates("key", keep="last")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce", format="mixed")
    return df.dropna(subset=["published_at"]).reset_index(drop=True)


def lexicon_as_label(text: str) -> dict:
    from trader.realtime.news_reflex import classify

    e = classify(text)
    syms = [s.split("/")[0] for s in e.symbols if s != "*"]
    return {"kind": e.kind, "sentiment": e.sentiment, "severity": e.severity, "symbols": syms,
            "relevant": e.kind != "other" or e.sentiment != 0}


def agreement(pred: list[dict], truth: pd.DataFrame) -> dict:
    t = truth.reset_index(drop=True)
    rel_ok = np.mean([p["relevant"] == r for p, r in zip(pred, t["relevant"])])
    mask = t["relevant"].to_numpy()
    kind_ok = np.mean([p["kind"] == k for p, k, m in zip(pred, t["kind"], mask) if m])
    sent_ok = np.mean([sent_class(p["sentiment"]) == sent_class(s) for p, s, m in zip(pred, t["sentiment"], mask) if m])
    sev_err = np.mean([abs(p["severity"] - s) for p, s, m in zip(pred, t["severity"], mask) if m])
    hi = t["severity"] >= 0.7  # o que importa para o reflexo: notícia grave
    caught = np.mean([p["severity"] >= 0.6 for p, h in zip(pred, hi) if h]) if hi.any() else float("nan")
    return {"relevante": rel_ok, "tipo": kind_ok, "direção": sent_ok, "erro_gravidade": sev_err,
            "graves_pegas": caught, "n_graves": int(hi.sum())}


def train(data: Path, models: Path, split: str = "2026-01-01") -> dict:
    df = load_labels(data)
    tr, te = df[df["published_at"] < split], df[df["published_at"] >= split]
    model = NewsModel().fit(tr)
    pred = model.predict(te["text"].tolist())
    lex = [lexicon_as_label(t) for t in te["text"]]
    t0 = time.perf_counter()
    for t in te["text"].head(200):
        model.predict([t])
    per_item_ms = (time.perf_counter() - t0) / min(200, len(te)) * 1000
    report = {"treino": len(tr), "teste": len(te), "modelo_vs_qwen": agreement(pred, te),
              "palavras_chave_vs_qwen": agreement(lex, te), "ms_por_manchete": round(per_item_ms, 3)}
    final = NewsModel().fit(df)  # o modelo em uso aprende com tudo
    # só vira o reflexo em uso se for melhor que chutar a classe mais comum e pegar notícias graves;
    # senão fica salvo como candidato (o reflexo continua nas palavras-chave)
    m = report["modelo_vs_qwen"]
    majority = max(te["relevant"].mean(), 1 - te["relevant"].mean())
    approved = m["relevante"] > majority + 0.05 and m["graves_pegas"] >= 0.5
    report["aprovado"] = bool(approved)
    path = models / (MODEL_FILE.name if approved else "news_reflex_candidate.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(final))
    report["salvo"] = str(path)
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)
    rep = train(data, data / "models")
    print(json.dumps(rep, ensure_ascii=False, indent=2, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
