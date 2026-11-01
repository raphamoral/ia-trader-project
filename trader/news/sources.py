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
    # imprensa geral de referência
    "cnn.com": 1, "bbc.com": 1, "bbc.co.uk": 1, "nytimes.com": 1, "theguardian.com": 1, "washingtonpost.com": 1,
    "straitstimes.com": 1, "elpais.com": 1, "lemonde.fr": 1, "spiegel.de": 1, "zeit.de": 1, "nhk.or.jp": 1,
    "koreaherald.com": 1, "koreatimes.co.kr": 1, "japantimes.co.jp": 1, "abc.net.au": 1, "afr.com": 1,
    "cnnbrasil.com.br": 1, "uol.com.br": 1, "oglobo.globo.com": 1, "poder360.com.br": 1, "moneytimes.com.br": 2,
    # dados de mercado e imprensa financeira secundária
    "finanzen.net": 2, "marketscreener.com": 2, "businessinsider.com": 2, "nasdaq.com": 2, "investing.com": 2,
    "fxstreet.com": 2, "kitco.com": 2, "mining.com": 2, "oilprice.com": 2, "seekingalpha.com": 2,
    "finance.yahoo.com": 2, "fortune.com": 2, "forbes.com": 2, "axios.com": 2, "politico.com": 2,
    # imprensa cripto estabelecida
    "coindesk.com": 2, "theblock.co": 2, "decrypt.co": 2, "blockworks.co": 2, "dlnews.com": 2,
    "cointelegraph.com": 2, "coinpost.jp": 2, "blockmedia.co.kr": 2, "btc-echo.de": 2,
    "criptonoticias.com": 2, "portaldobitcoin.uol.com.br": 2,
}

# relevante mas fora da lista: guardado (o backfill não se repete barato), fora das features
# por padrão; a reputação pode promover a fonte mais tarde
UNLISTED_TIER = 9
TIER_WEIGHT = {1: 1.0, 2: 0.6, UNLISTED_TIER: 0.0}


def tier_of(domain: str) -> int | None:
    """Nível da fonte, aceitando subdomínios (ex.: www.reuters.com). None = fora da lista."""
    domain = domain.lower().strip().removeprefix("www.")
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        tier = TIERS.get(".".join(parts[i:]))
        if tier is not None:
            return tier
    return None
