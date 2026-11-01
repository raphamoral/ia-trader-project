import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from trader.blacknews import alt
from trader.blacknews.sources import PoliteClient

T0 = pd.Timestamp("2026-01-01", tz="UTC")


def client(routes, log=None):
    def fetch(url, headers):
        if log is not None:
            log.append((url, headers.get("User-Agent")))
        body = routes.get(url)
        return (200, body, {}) if body is not None else (404, b"", {})
    return PoliteClient(fetch=fetch, sleep=lambda s: None)


# portão de termos


def test_gate_blocks_refused_sources_and_foreign_hosts():
    log = []
    c = client({}, log)
    for name in ("whale_alert", "polymarket", "kalshi", "coingecko", "youtube", "b3_lending"):
        with pytest.raises(alt.AltSourceNotAllowed):
            alt.fetch(c, name, f"https://{alt.POLICIES[name].host}/x")
    with pytest.raises(alt.AltSourceNotAllowed):
        alt.fetch(c, "fng", "https://evil.example/fng")
    assert log == []
    for p in alt.POLICIES.values():
        assert p.reason and p.checked


def test_polite_rate_and_contact_user_agent():
    log = []
    c = client({"https://api.github.com/repos/bitcoin/bitcoin/releases?per_page=100&page=1": b"[]"}, log)
    alt.fetch(c, "dev", "https://api.github.com/repos/bitcoin/bitcoin/releases?per_page=100&page=1")
    assert c.min_interval["api.github.com"] >= 60  # 60 req/h sem autenticação
    assert log[0][1] == alt.USER_AGENT  # Wikimedia/GitHub pedem contato: TRADER_CONTACT_EMAIL


# fng


def fng_body(values):
    data = [{"value": str(v), "timestamp": str(int((T0 + pd.Timedelta(days=i)).timestamp()))} for i, v in enumerate(values)]
    return json.dumps({"data": data[::-1]}).encode()


def test_fng_visible_next_day_and_causal():
    c = client({"https://api.alternative.me/fng/?limit=0&format=json": fng_body(range(10, 30))})
    df = alt.fetch_fng(c)
    idx = pd.date_range(T0, T0 + pd.Timedelta(days=19), freq="6h", tz="UTC")
    f = alt.fng_features(df, idx)
    assert np.isnan(f.loc[T0 + pd.Timedelta(hours=18), "black_alt_fng_value"])  # valor de D só em D+1
    assert f.loc[T0 + pd.Timedelta(days=1), "black_alt_fng_value"] == 10
    assert f.loc[T0 + pd.Timedelta(days=8), "black_alt_fng_chg_7d"] == 7
    cut = T0 + pd.Timedelta(days=5)
    part = alt.fng_features(df[df["visible_at"] <= cut], idx)
    pd.testing.assert_frame_equal(f[idx <= cut], part[idx <= cut])


# wiki


def test_wiki_zscore_uses_only_past_days():
    days = pd.date_range(T0, periods=60, freq="D", tz="UTC")
    views = [100.0] * 59 + [10_000.0]
    df = alt.frame([("wiki", "BTC|en.wikipedia|Bitcoin", d, d + pd.Timedelta(days=1, hours=6), "views", v)
                    for d, v in zip(days, views)])
    idx = pd.DatetimeIndex([days[-1] + pd.Timedelta(hours=12), days[-1] + pd.Timedelta(days=1, hours=7)])
    f = alt.wiki_features(df, idx, "BTC")
    assert f["black_alt_wiki_views_chg_1d"].iloc[0] == pytest.approx(0.0)  # pico ainda invisível
    assert f["black_alt_wiki_views_chg_1d"].iloc[1] > 4


# derivativos


def test_deriv_parse_and_features_causal():
    ts = pd.date_range(T0, periods=400, freq="5min")
    csv = pd.DataFrame({"create_time": ts.strftime("%Y-%m-%d %H:%M:%S"), "symbol": "BTCUSDT",
                        "sum_open_interest": 1.0, "sum_open_interest_value": np.linspace(1e9, 2e9, 400),
                        "count_toptrader_long_short_ratio": 1.5, "sum_toptrader_long_short_ratio": 1.1,
                        "count_long_short_ratio": 2.0, "sum_taker_long_short_vol_ratio": np.r_[np.ones(399), 5.0]}).to_csv(index=False)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BTCUSDT-metrics-2026-01-01.csv", csv)
    url = "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-01-01.zip"
    df = alt.fetch_deriv_day(client({url: buf.getvalue()}), T0)
    assert set(df["metric"]) == {"oi_usd", "top_ls", "ls", "taker"}
    last = ts[-1]
    idx = pd.DatetimeIndex([last, last + pd.Timedelta(minutes=5)])
    f = alt.deriv_features(df, idx)
    assert f["black_alt_deriv_taker"].iloc[0] == 1.0 and f["black_alt_deriv_taker"].iloc[1] == 5.0
    assert f["black_alt_deriv_oi_chg_1h"].iloc[1] > 0
    stale = alt.deriv_features(df, pd.DatetimeIndex([last + pd.Timedelta(days=30)]))
    assert stale.isna().all().all()  # dado velho não é repetido


# GitHub


def test_dev_counts_only_visible_stable_releases():
    rows = [("dev", "BTC|bitcoin/bitcoin", T0 + pd.Timedelta(days=d), T0 + pd.Timedelta(days=d, minutes=5), "release", v)
            for d, v in [(0, 1.0), (2, 0.0), (3, 1.0), (20, 1.0)]]
    f = alt.dev_features(alt.frame(rows), pd.DatetimeIndex([T0 + pd.Timedelta(days=4), T0 + pd.Timedelta(days=3)]), "BTC")
    assert f["black_alt_dev_releases_7d"].tolist() == [2.0, 1.0]  # pré-release fora; a de d=3 só após +5 min


# CVM: data de protocolo, nunca a data da operação; reapresentação só vale depois de protocolada


def cvm_zip(head_rows, mov_rows):
    head = "CNPJ_Companhia;Nome_Companhia;Data_Referencia;Versao;Codigo_CVM;Categoria;Tipo;Data_Entrega\n" + "\n".join(head_rows)
    mov = ("CNPJ_Companhia;Nome_Companhia;Data_Referencia;Versao;Tipo_Empresa;Empresa;Tipo_Cargo;Tipo_Movimentacao;"
           "Descricao_Movimentacao;Tipo_Operacao;Tipo_Ativo;Caracteristica_Valor_Mobiliario;Intermediario;"
           "Data_Movimentacao;Quantidade;Preco_Unitario;Volume\n") + "\n".join(mov_rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("vlmo_cia_aberta_2026.csv", head.encode("latin-1"))
        z.writestr("vlmo_cia_aberta_con_2026.csv", mov.encode("latin-1"))
    return buf.getvalue()


def test_cvm_point_in_time_by_filing_date_and_versions():
    pb = "33.000.167/0001-01;PETROLEO BRASILEIRO S.A. PETROBRAS"
    body = cvm_zip(
        [f"{pb};2026-01-01;1;9512;x;y;2026-02-10", f"{pb};2026-01-01;2;9512;x;y;2026-03-20"],
        [f"{pb};2026-01-01;1;Companhia;P;Diretoria;Compra à vista;;Crédito;Ações;PN;;2026-01-05;100;30;3000",
         f"{pb};2026-01-01;1;Companhia;P;Diretoria;Saldo Inicial;;Crédito;Ações;PN;;;999;;",
         f"{pb};2026-01-01;2;Companhia;P;Diretoria;Venda à vista;;Débito;Ações;PN;;2026-01-05;100;30;5000"],
    )
    df = alt.parse_cvm(body)
    assert set(df["value"]) == {3000.0, -5000.0} and (df["visible_at"] >= pd.Timestamp("2026-02-11", tz="UTC")).all()
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-01-20", "2026-02-12", "2026-03-22"], utc=True))
    f = alt.cvm_features(df, idx, "PETR4")
    assert f["black_alt_cvm_net_volume_30d"].tolist() == [0.0, 3000.0, -5000.0]  # antes do protocolo: nada


# braços e reputação


def test_alt_arms_one_per_family():
    cols = ["black_alt_fng_value", "black_alt_wiki_views_z", "black_alt_deriv_taker"]
    arms = alt.alt_arms(["news_a"], cols)
    assert set(arms) == {"W", "W+fng", "W+wiki", "W+deriv", "W+alt"}
    assert arms["W+wiki"] == ["news_a", "black_alt_wiki_views_z"]
    assert ("W+deriv", "W") in alt.alt_pairs(arms)


def test_alt_features_from_routes_families_by_asset():
    idx = pd.date_range(T0, periods=3, freq="D", tz="UTC")
    btc = alt.alt_features_from({}, "BTC", idx)
    petr = alt.alt_features_from({}, "PETR4", idx)
    assert any(c.startswith("black_alt_deriv_") for c in btc) and not any(c.startswith("black_alt_cvm_") for c in btc)
    assert any(c.startswith("black_alt_cvm_") for c in petr) and not any(c.startswith("black_alt_fng_") for c in petr)
    assert btc.index.equals(idx)


def test_alt_outcomes_use_only_past_signal():
    idx = pd.date_range(T0, periods=24 * 90, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    sig = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    feats = pd.DataFrame({"black_alt_fng_chg_7d": sig})
    prices = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    outs = alt.alt_outcomes(feats, prices)
    assert len(outs) > 0 and (outs["source"] == "alt:fng").all()
    # mudar o sinal DEPOIS de t não muda o desfecho registrado em t
    t = outs["resolved_at"].iloc[0] - pd.Timedelta(days=1)
    changed = feats.copy()
    changed.loc[changed.index > t, "black_alt_fng_chg_7d"] = 0.0
    outs2 = alt.alt_outcomes(changed, prices)
    first = outs2[outs2["item_id"] == outs["item_id"].iloc[0]]
    assert len(first) == 1 and bool(first["success"].iloc[0]) == bool(outs["success"].iloc[0])


def test_cvm_ticker_handles_accents():
    assert alt.cvm_ticker("PETRÓLEO BRASILEIRO S.A. - PETROBRAS") == "PETR4"
    assert alt.cvm_ticker("ITAÚ UNIBANCO HOLDING S.A.") == "ITUB4"
    assert alt.cvm_ticker("PETRO RIO JAGUAR PETRÓLEO S.A.") is None
