"""Dados alternativos (só APIs e bases públicas oficiais, termos verificados em 2026-09-29).

Cada fonte vira features `black_alt_<fonte>_*`, causais, no mesmo contrato de `black_features`
(mesmo índice, instante de decisão = t, só dados com visible_at <= t).

Visibilidade por fonte (sempre conservadora):
- fng (alternative.me): valor do dia D → visível em D + 1 dia (não sabemos a hora exata da publicação);
- wiki (pageviews): dia D → D + 1 dia + 6 h (a Wikimedia publica o dia fechado horas depois);
- deriv (data.binance.vision, métricas de 5 min): create_time + 5 min;
- dev (GitHub releases): published_at + 5 min;
- cvm (formulários de negociação de administradores, art. 11 da Res. CVM 44 / antiga Instr. 358):
  Data_Entrega (data de protocolo) + 1 dia. Nunca a data da operação. Reapresentações só
  substituem a versão anterior a partir da própria Data_Entrega.

Mesmos limites do monitor de segurança: portão de termos antes de qualquer requisição, sem
login/conta, sem contornar bloqueio.
"""
from __future__ import annotations

import io
import json
import os
import re
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from trader.blacknews.schema import utc64
from trader.blacknews.sources import PoliteClient
from trader.llm import load_env_file
from trader.storage.atomic import replace_with_retry, unique_tmp

load_env_file()
_CONTACT = os.environ.get("TRADER_CONTACT_EMAIL", "").strip()  # fontes como a SEC pedem contato no User-Agent
USER_AGENT = f"AI-trader-project/blacknews (personal research; non-commercial{'; ' + _CONTACT if _CONTACT else ''})"
ALT_COLUMNS = ["source", "key", "ts", "visible_at", "metric", "value"]


class AltSourceNotAllowed(Exception):
    pass


@dataclass(frozen=True)
class AltPolicy:
    source: str
    host: str
    allowed: bool
    reason: str
    depth: str
    min_interval: float = 1.0
    checked: str = "2026-09-29"


POLICIES: dict[str, AltPolicy] = {
    p.source: p
    for p in [
        AltPolicy("fng", "api.alternative.me", True,
                  "API pública sem chave, 60 req/min, uso livre (inclusive comercial) com atribuição", "fev/2018 → hoje, diário"),
        AltPolicy("wiki", "wikimedia.org", True,
                  "API REST oficial de pageviews; exige User-Agent com contato", "jul/2015 → hoje, diário"),
        AltPolicy("deriv", "data.binance.vision", True,
                  "base pública de dados de mercado da Binance (métricas de futuros USDT-M)", "dez/2021 → hoje, 5 min"),
        AltPolicy("deriv_api", "fapi.binance.com", True,
                  "endpoints públicos de dados de mercado (futures/data/*); só ~30 dias de histórico", "~30 dias"),
        AltPolicy("dev", "api.github.com", True,
                  "API pública; sem autenticação = 60 req/h (respeitado com intervalo de 65 s)", "anos (desde a 1ª release)",
                  min_interval=65.0),
        AltPolicy("cvm", "dados.cvm.gov.br", True,
                  "Portal Dados Abertos CVM; robots: Crawl-Delay 10, /dados/ liberado", "2018 → hoje, mensal",
                  min_interval=10.0),
        AltPolicy("whale_alert", "api.whale-alert.io", False, "sem plano gratuito de API (pago a partir de US$29,95/mês)", "—"),
        AltPolicy("exchange_flows", "-", False, "nenhuma fonte pública gratuita e oficial de fluxo de entrada/saída das exchanges", "—"),
        AltPolicy("polymarket", "gamma-api.polymarket.com", False,
                  "inacessível desta rede (tempo esgotado); em mar/2026 o regulador declarou mercados de previsão não "
                  "autorizados no Brasil. Não contornamos o bloqueio", "—"),
        AltPolicy("kalshi", "external-api.kalshi.com", False, "inacessível desta rede (tempo esgotado); não contornamos", "—"),
        AltPolicy("coingecko", "api.coingecko.com", False, "agora exige chave Demo, que exige conta (fora dos limites)", "—"),
        AltPolicy("youtube", "www.googleapis.com", False, "chave exige conta Google (fora dos limites)", "—"),
        AltPolicy("b3_lending", "arquivos.b3.com.br", False,
                  "posições de aluguel só pelo Boletim Diário (desde 25/09/2023) via endpoint interno não documentado; "
                  "API oficial exige cadastro. Pendente: baixar os arquivos manualmente, como o COTAHIST", "—"),
    ]
}


def gate(source: str, url: str) -> AltPolicy:
    policy = POLICIES.get(source)
    if policy is None or not policy.allowed:
        raise AltSourceNotAllowed(f"{source}: {policy.reason if policy else 'fonte não verificada'}")
    if urllib.parse.urlparse(url).netloc != policy.host:
        raise AltSourceNotAllowed(f"{source}: host fora da política ({url})")
    return policy


def fetch(client: PoliteClient, source: str, url: str) -> bytes | None:
    policy = gate(source, url)
    client.min_interval[policy.host] = max(client.min_interval.get(policy.host, 0.0), policy.min_interval)
    return client.get(url, {"User-Agent": USER_AGENT})


def frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=ALT_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["visible_at"] = pd.to_datetime(df["visible_at"], utc=True)
    df["value"] = pd.to_numeric(df["value"])
    return df


class AltStore:
    def __init__(self, root: Path | str = Path("data") / "news_black") -> None:
        self.dir = Path(root) / "alt"

    def path(self, source: str) -> Path:
        return self.dir / f"{source}.parquet"

    def write(self, source: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.path(source)
        before = 0
        if p.exists():
            old = pd.read_parquet(p)
            before = len(old)
            df = pd.concat([old, df])
        # a primeira observação visível manda (não reescrevemos o passado com revisões)
        df = df.sort_values("visible_at").drop_duplicates(["key", "ts", "metric"], keep="first")
        df = df.sort_values(["key", "metric", "ts"]).reset_index(drop=True)
        tmp = unique_tmp(p)
        df.to_parquet(tmp, index=False)
        replace_with_retry(tmp, p)
        return len(df) - before

    def read(self, source: str) -> pd.DataFrame:
        p = self.path(source)
        return pd.read_parquet(p) if p.exists() else frame([])


# conectores


def fetch_fng(client: PoliteClient) -> pd.DataFrame:
    body = fetch(client, "fng", "https://api.alternative.me/fng/?limit=0&format=json")
    rows = []
    for d in json.loads(body)["data"] if body else []:
        day = pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC")
        rows.append(("fng", "crypto", day, day + pd.Timedelta(days=1), "fng", float(d["value"])))
    return frame(rows)


WIKI_ARTICLES = {
    "BTC": [("en.wikipedia", "Bitcoin"), ("en.wikipedia", "Cryptocurrency"), ("en.wikipedia", "Binance"),
            ("en.wikipedia", "Coinbase")],
    "PETR4": [("pt.wikipedia", "Petrobras")],
    "VALE3": [("pt.wikipedia", "Vale_S.A.")],
    "IBOV": [("pt.wikipedia", "Ibovespa")],
}


def fetch_wiki(client: PoliteClient, start: str = "20150701", end: str | None = None) -> pd.DataFrame:
    end = end or pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    rows = []
    for asset, articles in WIKI_ARTICLES.items():
        for project, article in articles:
            url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                   f"{project}/all-access/user/{urllib.parse.quote(article, safe='')}/daily/{start}/{end}")
            body = fetch(client, "wiki", url)
            for it in json.loads(body).get("items", []) if body else []:
                day = pd.Timestamp(it["timestamp"][:8], tz="UTC")
                rows.append(("wiki", f"{asset}|{project}|{article}", day, day + pd.Timedelta(days=1, hours=6),
                             "views", float(it["views"])))
    return frame(rows)


DERIV_METRICS = {
    "sum_open_interest_value": "oi_usd",
    "count_toptrader_long_short_ratio": "top_ls",
    "count_long_short_ratio": "ls",
    "sum_taker_long_short_vol_ratio": "taker",
}


def fetch_deriv_day(client: PoliteClient, day: pd.Timestamp, symbol: str = "BTCUSDT") -> pd.DataFrame:
    url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"
    body = fetch(client, "deriv", url)
    if not body:
        return frame([])
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]))
    ts = pd.to_datetime(raw["create_time"], utc=True)
    rows = []
    for col, metric in DERIV_METRICS.items():
        if col in raw:
            rows += [("deriv", symbol, t, t + pd.Timedelta(minutes=5), metric, v) for t, v in zip(ts, raw[col])]
    return frame(rows)


GITHUB_REPOS = {"BTC": ["bitcoin/bitcoin", "ethereum/go-ethereum", "anza-xyz/agave", "XRPLF/rippled"]}


def fetch_dev(client: PoliteClient, pages: int = 1) -> pd.DataFrame:
    rows = []
    for asset, repos in GITHUB_REPOS.items():
        for repo in repos:
            for page in range(1, pages + 1):
                body = fetch(client, "dev", f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}")
                releases = json.loads(body) if body else []
                for r in releases:
                    if r.get("draft") or not r.get("published_at"):
                        continue
                    t = pd.Timestamp(r["published_at"]).tz_convert("UTC")
                    rows.append(("dev", f"{asset}|{repo}", t, t + pd.Timedelta(minutes=5), "release",
                                 0.0 if r.get("prerelease") else 1.0))
                if len(releases) < 100:
                    break
    return frame(rows)


CVM_TICKERS = {
    r"PETROLEO BRASILEIRO": "PETR4", r"^VALE S\.?A": "VALE3", r"ITAU UNIBANCO HOLDING": "ITUB4",
    r"BANCO BRADESCO": "BBDC4", r"BANCO DO BRASIL": "BBAS3", r"^AMBEV": "ABEV3", r"^WEG S\.?A": "WEGE3",
    r"^B3 S\.?A": "B3SA3",
}
_NOT_TRADES = {"Saldo Inicial", "Saldo Final"}


def cvm_ticker(name: str) -> str | None:
    import unicodedata

    up = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().upper()
    return next((t for rx, t in CVM_TICKERS.items() if re.search(rx, up)), None)


def parse_cvm(zip_bytes: bytes) -> pd.DataFrame:
    """Uma linha por movimentação de administrador/controlador, com a data de protocolo (Data_Entrega)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        head = pd.read_csv(z.open(next(n for n in names if "_con_" not in n)), sep=";", encoding="latin-1", dtype=str)
        mov = pd.read_csv(z.open(next(n for n in names if "_con_" in n)), sep=";", encoding="latin-1", dtype=str)
    keys = ["CNPJ_Companhia", "Data_Referencia", "Versao"]
    mov = mov.merge(head[keys + ["Data_Entrega"]], on=keys, how="inner")
    mov = mov[~mov["Tipo_Movimentacao"].isin(_NOT_TRADES)]
    mov["ticker"] = mov["Nome_Companhia"].map(cvm_ticker)
    mov = mov.dropna(subset=["ticker", "Data_Entrega"])
    vol = pd.to_numeric(mov["Volume"].str.replace(",", "."), errors="coerce").fillna(0.0)
    sign = np.where(mov["Tipo_Operacao"].str.startswith("Cr"), 1.0, np.where(mov["Tipo_Operacao"].str.startswith("D"), -1.0, 0.0))
    filed = pd.to_datetime(mov["Data_Entrega"], utc=True)
    out = pd.DataFrame({
        "source": "cvm",
        "key": mov["ticker"] + "|" + mov["Data_Referencia"] + "|v" + mov["Versao"],
        "ts": pd.to_datetime(mov["Data_Referencia"], utc=True),
        "visible_at": filed + pd.Timedelta(days=1),
        "metric": "net_volume_brl",
        "value": sign * vol,
    })
    # soma por formulário (uma linha por ticker × competência × versão)
    return out.groupby(["source", "key", "ts", "visible_at", "metric"], as_index=False)["value"].sum()[ALT_COLUMNS]


def fetch_cvm(client: PoliteClient, years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        body = fetch(client, "cvm", f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/VLMO/DADOS/vlmo_cia_aberta_{y}.zip")
        if body:
            frames.append(parse_cvm(body))
    return pd.concat(frames, ignore_index=True) if frames else frame([])


# features causais


def _asof(series_ts: pd.Series, values: pd.Series, index: pd.DatetimeIndex,
          max_age: pd.Timedelta | None = None) -> np.ndarray:
    """Último valor com visible_at <= t para cada t (NaN antes do primeiro, ou se mais velho que max_age)."""
    if len(series_ts) == 0:
        return np.full(len(index), np.nan)
    order = np.argsort(utc64(series_ts), kind="stable")
    vis = utc64(series_ts)[order]
    vals = values.to_numpy(dtype=float)[order]
    pos = np.searchsorted(vis, utc64(pd.Series(index)), "right") - 1
    out = np.full(len(index), np.nan)
    ok = pos >= 0
    if max_age is not None:
        t = utc64(pd.Series(index))
        ok &= (t - vis[np.maximum(pos, 0)]) <= np.timedelta64(int(max_age.total_seconds()), "s")
    out[ok] = vals[pos[ok]]
    return out


def fng_features(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    s = df[df["metric"] == "fng"].sort_values("ts")
    chg = s["value"] - s["value"].shift(7)  # usa só dias anteriores, visíveis antes
    return pd.DataFrame({
        "black_alt_fng_value": _asof(s["visible_at"], s["value"], index, pd.Timedelta(days=3)),
        "black_alt_fng_chg_7d": _asof(s["visible_at"], chg, index, pd.Timedelta(days=3)),
    }, index=index)


def wiki_features(df: pd.DataFrame, index: pd.DatetimeIndex, asset: str) -> pd.DataFrame:
    s = df[df["key"].astype(str).str.startswith(f"{asset}|")]
    cols = ["black_alt_wiki_views_z", "black_alt_wiki_views_chg_1d"]
    if s.empty:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    day = s.groupby("ts").agg(views=("value", "sum"), visible_at=("visible_at", "max")).sort_index()
    logv = np.log1p(day["views"])
    mean, std = logv.shift(1).rolling(28, min_periods=14).mean(), logv.shift(1).rolling(28, min_periods=14).std()
    return pd.DataFrame({
        cols[0]: _asof(day["visible_at"], (logv - mean) / std, index, pd.Timedelta(days=4)),
        cols[1]: _asof(day["visible_at"], logv.diff(), index, pd.Timedelta(days=4)),
    }, index=index)


def deriv_features(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    wide = df.pivot_table(index="ts", columns="metric", values="value").sort_index() if len(df) else pd.DataFrame()
    cols = ["black_alt_deriv_oi_chg_1h", "black_alt_deriv_oi_chg_24h", "black_alt_deriv_top_ls",
            "black_alt_deriv_ls", "black_alt_deriv_taker", "black_alt_deriv_taker_z"]
    if wide.empty:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    wide = wide.asfreq("5min")  # buracos viram NaN em vez de encurtar a janela
    vis = pd.Series(wide.index + pd.Timedelta(minutes=5), index=wide.index)
    oi = np.log(wide["oi_usd"].where(wide["oi_usd"] > 0))
    taker = wide["taker"]
    tz = (taker - taker.shift(1).rolling(288, min_periods=100).mean()) / taker.shift(1).rolling(288, min_periods=100).std()
    series = [oi.diff(12), oi.diff(288), wide.get("top_ls"), wide.get("ls"), taker, tz]
    stale = pd.Timedelta(minutes=30)  # sem dado recente = NaN (não repete valor antigo)
    return pd.DataFrame({c: _asof(vis, v, index, stale) if v is not None else np.nan for c, v in zip(cols, series)},
                        index=index)


def dev_features(df: pd.DataFrame, index: pd.DatetimeIndex, asset: str) -> pd.DataFrame:
    s = df[df["key"].astype(str).str.startswith(f"{asset}|") & (df["value"] == 1.0)].sort_values("visible_at")
    vis = utc64(s["visible_at"])
    t = utc64(pd.Series(index))
    week = np.timedelta64(7, "D")
    count = np.searchsorted(vis, t, "right") - np.searchsorted(vis, t - week, "right")
    last = np.searchsorted(vis, t, "right") - 1
    if len(vis):
        since = np.where(last >= 0, (t - vis[np.maximum(last, 0)]) / np.timedelta64(1, "D"), np.nan)
    else:
        since = np.full(len(t), np.nan)
    return pd.DataFrame({"black_alt_dev_releases_7d": count.astype(float), "black_alt_dev_days_since_release": since},
                        index=index)


def cvm_features(df: pd.DataFrame, index: pd.DatetimeIndex, ticker: str, window: pd.Timedelta = pd.Timedelta(days=30)) -> pd.DataFrame:
    """Fluxo líquido dos administradores em formulários PROTOCOLADOS nos últimos 30 dias até t.
    Em cada t, vale a versão mais recente de cada competência já protocolada em t."""
    s = df[df["key"].astype(str).str.startswith(f"{ticker}|")].copy()
    cols = ["black_alt_cvm_net_volume_30d", "black_alt_cvm_filings_30d"]
    if s.empty:
        return pd.DataFrame(0.0, index=index, columns=cols)
    s["ref"] = s["key"].str.rsplit("|", n=1).str[0]
    s = s.sort_values("visible_at")
    out = np.zeros((len(index), 2))
    vis = utc64(s["visible_at"])
    for i, t in enumerate(utc64(pd.Series(index))):
        known = s.iloc[: np.searchsorted(vis, t, "right")]
        if known.empty:
            continue
        latest = known.groupby("ref").tail(1)  # versão mais recente conhecida em t
        recent = latest[utc64(latest["visible_at"]) > t - np.timedelta64(int(window.total_seconds()), "s")]
        out[i] = (recent["value"].sum(), len(recent))
    return pd.DataFrame(out, index=index, columns=cols)


FAMILIES = ("fng", "wiki", "deriv", "dev", "cvm")
CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "XRP"}


def alt_features_from(stores: dict[str, pd.DataFrame], asset: str, index: pd.DatetimeIndex,
                      families: tuple[str, ...] = FAMILIES) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    parts = []
    crypto = asset in CRYPTO_ASSETS
    for fam in families:
        df = stores.get(fam, frame([]))
        if fam == "fng" and crypto:
            parts.append(fng_features(df, idx))
        elif fam == "wiki":
            parts.append(wiki_features(df, idx, "BTC" if crypto else asset))
        elif fam == "deriv" and crypto:
            parts.append(deriv_features(df, idx))
        elif fam == "dev" and crypto:
            parts.append(dev_features(df, idx, "BTC"))
        elif fam == "cvm" and not crypto:
            parts.append(cvm_features(df, idx, asset))
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=idx)
    out.index = index
    assert all(c.startswith("black_alt_") for c in out.columns)
    return out


def family_of(column: str) -> str:
    return column.removeprefix("black_alt_").split("_", 1)[0]


def alt_arms(white: list[str], alt_columns: list[str]) -> dict[str, list[str]]:
    """Um braço por família (W + família) e um com todas (W + alt), para medir cada fonte separadamente."""
    arms = {"W": list(white)}
    for fam in sorted({family_of(c) for c in alt_columns}):
        arms[f"W+{fam}"] = [*white, *[c for c in alt_columns if family_of(c) == fam]]
    arms["W+alt"] = [*white, *alt_columns]
    return arms


def alt_pairs(arms: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(a, "W") for a in arms if a != "W"]


# gancho de reputação: cada família ganha nota pela direção que o sinal previu


PRIMARY_SIGNAL = {
    "fng": "black_alt_fng_chg_7d", "wiki": "black_alt_wiki_views_z", "deriv": "black_alt_deriv_taker_z",
    "dev": "black_alt_dev_releases_7d", "cvm": "black_alt_cvm_net_volume_30d",
}


def alt_outcomes(features: pd.DataFrame, prices: pd.Series, horizon: pd.Timedelta = pd.Timedelta(days=1),
                 threshold: float = 1.0, step: str = "1D") -> pd.DataFrame:
    """Desfechos de direção no esquema do trader.news.reputation: quando o sinal primário da família
    está forte (|z| > threshold) num instante diário, acertou se o preço andou no mesmo sentido até t+horizon.
    `prices`: fechamentos indexados pelo instante em que ficam conhecidos."""
    from trader.news.reputation import OUTCOME_COLUMNS

    rows = []
    # amostra diária causal: em cada instante t, o último valor calculado em t ou antes (nunca o fim do dia)
    grid = pd.date_range(features.index.min().ceil(step), features.index.max(), freq=step)
    daily = features.reindex(features.index.union(grid)).ffill().reindex(grid)
    for fam, col in PRIMARY_SIGNAL.items():
        if col not in daily:
            continue
        sig = daily[col]
        z = (sig - sig.shift(1).rolling(60, min_periods=20).mean()) / sig.shift(1).rolling(60, min_periods=20).std()
        for t, v in z.dropna().items():
            if abs(v) <= threshold:
                continue
            p0 = prices[:t]
            p1 = prices[: t + horizon]
            if p0.empty or p1.index[-1] < t + horizon - pd.Timedelta(hours=1):
                continue
            move = p1.iloc[-1] / p0.iloc[-1] - 1
            rows.append({"source": f"alt:{fam}", "kind": "direction", "success": bool(np.sign(move) == np.sign(v)),
                         "resolved_at": t + horizon, "item_id": f"{fam}:{t:%Y%m%d}", "weight": 1.0})
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


def alt_reputation(features: pd.DataFrame, prices: pd.Series, horizon: pd.Timedelta = pd.Timedelta(days=1)):
    """Reputation (trader.news) por família alternativa, priori média (tier None), point-in-time:
    rep.weights(t, ["alt:fng", "alt:wiki", ...]) usa só desfechos resolvidos antes de t."""
    from trader.news.reputation import Reputation

    outs = alt_outcomes(features, prices, horizon)
    return Reputation(outs, tiers={f"alt:{f}": None for f in FAMILIES})
