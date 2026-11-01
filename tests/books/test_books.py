import hashlib
import json

import pymupdf as fitz
import numpy as np

from trader.books.agentic import AgenticLibrary
from trader.books.index import Library, book_title, build, pick_editions

RISK = ("Always use a stop loss. Position sizing and money management protect you after a losing streak. "
        "Risk no more than one percent per trade and respect your drawdown limit. ") * 6
ENTRY = ("A breakout above resistance with volume is a strong entry setup. Wait for the pullback to enter "
         "and place the trigger above the high of the setup bar. Fast average of 18 to 22 bars, slow of 45 to 55. ") * 6


def fake_embed(texts):
    """Vetores determinísticos: palavras viram dimensões (suficiente para ranquear por tema)."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 560, 800), text, fontsize=9)
    doc.save(path)


def test_index_search_with_page_citation_and_filters(tmp_path):
    src = tmp_path / "library"
    src.mkdir()
    make_pdf(src / "Day Trading Risk Manual 2010.pdf", [ENTRY, RISK])
    make_pdf(src / "Day Trading Risk Manual 2010 (1).pdf", [ENTRY, RISK])  # duplicata: ignorada
    stats = build(src, tmp_path / "index", embed=fake_embed, log=lambda m: None)
    assert stats["novos"] == 1 and stats["trechos"] >= 2
    lib = Library.load(tmp_path / "index")
    hits = lib.search("stop loss position sizing losing streak", k=1, topics=["risk"], embed=fake_embed)
    assert hits[0]["page"] == 2 and hits[0]["book"] == "Day Trading Risk Manual 2010"
    assert "day_trade" in hits[0]["styles"]
    again = build(src, tmp_path / "index", embed=fake_embed, log=lambda m: None)
    assert again["novos"] == 0  # incremental: nada a refazer


def test_edition_picking_prefers_pdf_and_drops_copies(tmp_path):
    for name in ["Book A 2001.pdf", "Book A 2001 (1).pdf", "Book A 2001.epub", "Book B.mobi"]:
        (tmp_path / name).write_bytes(b"x" * 10)
    picked = sorted(p.name for p in pick_editions(list(tmp_path.iterdir())))
    assert picked == ["Book A 2001.pdf", "Book B.mobi"] or picked == ["Book A 2001 (1).pdf", "Book B.mobi"]
    assert book_title(tmp_path / "Mastering X 2012 (1) - annotated.pdf") == "Mastering X 2012"


def test_agentic_search_plans_retries_once_and_caches(tmp_path):
    src = tmp_path / "library"
    src.mkdir()
    make_pdf(src / "Swing Trading Guide.pdf", [ENTRY, RISK])
    build(src, tmp_path / "index", embed=fake_embed, log=lambda m: None)
    lib = Library.load(tmp_path / "index")
    calls = []

    def post(url, payload, timeout):
        calls.append(payload)
        plan = {"styles": ["swing"], "topics": ["risk"], "query": "stop loss after losing streak position sizing"}
        return {"choices": [{"message": {"content": json.dumps(plan)}}]}

    agent = AgenticLibrary(lib, post=post, search_kwargs={"embed": fake_embed})
    first = agent("It lost 18% after several losing trades in a row")
    assert first and first[0]["page"] == 2 and 1 <= len(calls) <= 2
    n = len(calls)
    assert agent("It lost 18% after several losing trades in a row") == first and len(calls) == n  # cache


def test_distill_keeps_only_cited_rules_and_builds_priors(tmp_path):
    from trader.books.distill import distill, top_rules
    from trader.evolution.genome import random_genome
    import random as _r

    src = tmp_path / "library"
    src.mkdir()
    make_pdf(src / "Swing Trading Guide.pdf", [ENTRY, RISK])
    build(src, tmp_path / "index", embed=fake_embed, log=lambda m: None)
    lib = Library.load(tmp_path / "index")

    def post(url, payload, timeout):
        rules = {"rules": [
            {"rule": "Use a 20/50 moving average crossover.", "genes": {"ma_cross.fast": [18, 22], "ma_cross.slow": [45, 55]},
             "citations": ["[Swing Trading Guide p.1]"]},
            {"rule": "Invented rule without source.", "genes": {}, "citations": ["[Livro Inexistente p.9]"]},
        ]}
        return {"choices": [{"message": {"content": json.dumps(rules)}}]}

    playbook = distill(lib, post=post, search_kwargs={"embed": fake_embed}, log=lambda m: None)
    assert playbook["rules"] and all("Inexistente" not in r["rule"] + " ".join(r["citations"]) for r in playbook["rules"])
    assert playbook["priors"]["ma_cross.fast"]["low"] == 18
    assert "[Swing Trading Guide p.1]" in top_rules(playbook, 1)[0]
    rng = _r.Random(0)
    fast = [random_genome(rng, "ma_cross", priors=playbook["priors"], prior_weight=1.0).params["fast"] for _ in range(30)]
    assert all(18 <= f <= 22 for f in fast)


def test_bad_chunk_is_skipped_not_fatal():
    from trader.books.index import clean, embed_rows

    assert clean("ok\x00 \ud800 fim") == "ok   fim" or "\x00" not in clean("ok\x00 fim")
    rows = [{"text": t, "page": i} for i, t in enumerate(["bom", "RUIM", "bom de novo"])]

    def picky(texts):
        if any("RUIM" in t for t in texts):
            raise ValueError("400 Bad Request")
        return fake_embed(texts)

    kept, vecs = embed_rows(rows, picky, batch=3, log=lambda m: None)
    assert [r["text"] for r in kept] == ["bom", "bom de novo"] and vecs.shape[0] == 2


def test_grounded_genes_rejects_schema_echo_invented_and_unknown():
    from trader.books.distill import grounded_genes

    text = "Use a 20-day breakout and risk no more than 2% per trade. A 50 and 200 day average."
    got = grounded_genes({
        "breakout.lookback": [20, 20],     # está no texto: fica
        "ma_cross.fast": [50, 50],          # está no texto: fica
        "trend_filter": [20, 300],          # faixa do esquema copiada: sai
        "cooldown": [30, 30],               # número inventado: sai
        "position_size": [0.02, 0.02],      # gene que não existe: sai
    }, text)
    assert got == {"breakout.lookback": [20.0, 20.0], "ma_cross.fast": [50.0, 50.0]}


def test_distill_retries_with_fewer_passages(tmp_path, monkeypatch):
    from trader.books import distill as D

    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    src = tmp_path / "library"
    src.mkdir()
    make_pdf(src / "Swing Trading Guide.pdf", [ENTRY, RISK] * 4)
    build(src, tmp_path / "index", embed=fake_embed, log=lambda m: None)
    lib = Library.load(tmp_path / "index")
    sizes = []

    def post(url, payload, timeout):
        n = payload["messages"][0]["content"].count("[Swing Trading Guide p.")
        sizes.append(n)
        if len(sizes) < 3:
            raise ConnectionError("ollama ocupado")
        return {"choices": [{"message": {"content": json.dumps({"rules": []})}}]}

    D.distill(lib, post=post, search_kwargs={"embed": fake_embed}, log=lambda m: None)
    assert sizes[:3] == sorted(sizes[:3], reverse=True) and sizes[2] <= 3
