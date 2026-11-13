"""Descompacta o JSON gerado por devtools_dump.js em pastas navegáveis.

Uso:
    python -I scripts/devtools_dump/unpack_dump.py dump_xxx.json [pasta_saida]

Saída:
    files/<host>/<caminho>     conteúdo de cada recurso (JS, CSS, HTML, maps...)
    sources/<...>              fontes originais reconstruídas dos source maps
    inline/scripts_NNN.js      scripts inline;  inline/styles_NNN.css
    dom_live.html              DOM no momento do save
    network.csv                timing de todos os recursos (aba Network)
    live_http.json             fetch/XHR capturados ao vivo, com corpo
    live_ws.jsonl              mensagens WebSocket;  console.jsonl;  errors.json
    storage.json, globals.json, meta.json, links.txt
"""
from __future__ import annotations

import base64
import csv
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

_BAD = re.compile(r'[<>:"|?*\x00-\x1f]')


def safe_rel(*parts: str) -> Path:
    """Caminho relativo seguro (sem '..', sem drive, sem chars inválidos no Windows)."""
    out: list[str] = []
    for part in parts:
        for seg in re.split(r"[\\/]+", part):
            seg = _BAD.sub("_", unquote(seg)).strip(" .")
            if seg and seg not in (".", ".."):
                out.append(seg[:150])
    return Path(*out) if out else Path("_")


def url_to_path(url: str) -> Path:
    u = urlsplit(url)
    path = u.path or "/"
    if path.endswith("/"):
        path += "index.html"
    rel = safe_rel(u.netloc or "_", path)
    if u.query:
        rel = rel.with_name(f"{rel.stem}__{_BAD.sub('_', u.query)[:60]}{rel.suffix}")
    return rel


def write(root: Path, rel: Path, data: str | bytes) -> None:
    dest = (root / rel).resolve()
    if root.resolve() not in dest.parents:
        raise ValueError(f"caminho fora da saída: {rel}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        dest.write_bytes(data)
    else:
        dest.write_text(data, encoding="utf-8", errors="replace")


def dump_json(root: Path, name: str, obj) -> None:
    write(root, Path(name), json.dumps(obj, ensure_ascii=False, indent=2))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    d = json.loads(src.read_text(encoding="utf-8"))

    dump_json(out, "meta.json", {"meta": d.get("meta"), "navigation": d.get("navigation"), "metaTags": d.get("metaTags")})
    write(out, Path("dom_live.html"), d.get("domLive", ""))
    write(out, Path("links.txt"), "\n".join(d.get("links", [])))

    for s in d.get("inlineScripts", []):
        ext = "json" if "json" in (s.get("type") or "") else "js"
        write(out, Path("inline") / f"scripts_{s['i']:03d}.{ext}", s.get("text") or "")
    for s in d.get("inlineStyles", []):
        write(out, Path("inline") / f"styles_{s['i']:03d}.css", s.get("text") or "")

    n_files = n_src = 0
    index = []
    for url, r in d.get("resources", {}).items():
        rel = Path("files") / url_to_path(url)
        index.append({"url": url, "file": str(rel), "status": r.get("status"), "type": r.get("type"), "error": r.get("error"), "skipped": r.get("skipped")})
        if r.get("text") is not None:
            write(out, rel, r["text"]); n_files += 1
        elif r.get("base64"):
            write(out, rel, base64.b64decode(r["base64"])); n_files += 1
        # source map: reconstrói fontes originais
        if r.get("text") and (url.split("?")[0].endswith(".map") or r["text"].lstrip().startswith('{"version"')):
            try:
                sm = json.loads(r["text"])
            except ValueError:
                continue
            for name, content in zip(sm.get("sources", []), sm.get("sourcesContent") or []):
                if content is not None:
                    name = re.sub(r"^[a-z]+:/+", "", name)  # webpack:///, file:// ...
                    write(out, Path("sources") / safe_rel(urlsplit(url).netloc, name), content); n_src += 1
    dump_json(out, "resources_index.json", index)

    with (out / "network.csv").open("w", newline="", encoding="utf-8") as f:
        rows = d.get("network", [])
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    live = d.get("live", {})
    dump_json(out, "live_http.json", live.get("http", []))
    dump_json(out, "errors.json", live.get("errors", []))
    dump_json(out, "live_sse_postmessage.json", {"sse": live.get("sse", []), "postMessage": live.get("messages", [])})
    for name, key in (("live_ws.jsonl", "ws"), ("console.jsonl", "console")):
        write(out, Path(name), "\n".join(json.dumps(x, ensure_ascii=False) for x in live.get(key, [])))
    dump_json(out, "storage.json", d.get("storage", {}))
    dump_json(out, "globals.json", d.get("globals", {}))

    print(f"ok -> {out}")
    print(f"  arquivos: {n_files}  fontes de source map: {n_src}  req ao vivo: {len(live.get('http', []))}  "
          f"ws: {len(live.get('ws', []))}  console: {len(live.get('console', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
