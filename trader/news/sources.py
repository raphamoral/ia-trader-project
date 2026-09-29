"""Lista branca de fontes confiáveis, com nível de confiabilidade por domínio.

Nível 1: agências globais, bancos centrais, bolsas e grande imprensa financeira.
Nível 2: imprensa cripto estabelecida (redação própria, correções públicas).
Tudo que não está aqui é descartado na ingestão: fazendas de conteúdo, blogs
anônimos, agregadores e redes sociais.
"""
from __future__ import annotations

TIERS: dict[str, int] = {
    # agências e imprensa financeira global
    "reuters.com": 1, "bloomberg.com": 1, "apnews.com": 1, "ft.com": 1, "wsj.com": 1,
    "cnbc.com": 1, "economist.com": 1, "marketwatch.com": 1, "barrons.com": 1,
    # bancos centrais, reguladores e bolsas
    "bcb.gov.br": 1, "federalreserve.gov": 1, "ecb.europa.eu": 1, "sec.gov": 1, "cvm.gov.br": 1,
    "b3.com.br": 1, "binance.com": 1,
    # Brasil
    "valor.globo.com": 1, "infomoney.com.br": 1, "estadao.com.br": 1, "folha.uol.com.br": 1,
    "exame.com": 1, "g1.globo.com": 1, "agenciabrasil.ebc.com.br": 1, "bloomberglinea.com.br": 1,
    # Ásia e Europa
    "asia.nikkei.com": 1, "nikkei.com": 1, "caixinglobal.com": 1, "scmp.com": 1, "en.yna.co.kr": 1,
    "yna.co.kr": 1, "handelsblatt.com": 1, "faz.net": 1, "expansion.com": 1, "cincodias.elpais.com": 1,
    "lesechos.fr": 1, "ilsole24ore.com": 1,
    # imprensa cripto estabelecida
    "coindesk.com": 2, "theblock.co": 2, "decrypt.co": 2, "blockworks.co": 2, "dlnews.com": 2,
    "cointelegraph.com": 2, "coinpost.jp": 2, "blockmedia.co.kr": 2, "btc-echo.de": 2,
    "criptonoticias.com": 2, "portaldobitcoin.uol.com.br": 2,
}

TIER_WEIGHT = {1: 1.0, 2: 0.6}


def tier_of(domain: str) -> int | None:
    """Nível da fonte, aceitando subdomínios (ex.: www.reuters.com). None = fora da lista."""
    domain = domain.lower().strip().removeprefix("www.")
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        tier = TIERS.get(".".join(parts[i:]))
        if tier is not None:
            return tier
    return None
