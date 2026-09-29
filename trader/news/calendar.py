"""Calendário macro: horários agendados de Copom, FOMC e payroll.

Eventos agendados são públicos com antecedência (o calendário do Copom sai no
ano anterior; o do FOMC também), então usar o horário agendado antes do evento
não é olhar o futuro. O *resultado* do evento só existe depois do horário.

- Copom: API aberta do BCB (histórico de decisões). Comunicado às 18:30 de Brasília.
- FOMC: página oficial do Fed. Comunicado às 14:00 de Nova York.
- Payroll (EUA): aproximação "primeira sexta-feira do mês, 08:30 NY". O BLS
  bloqueia acesso automatizado à agenda oficial; há exceções à regra (ex.: quando
  o dia 1º cai numa sexta), marcadas como approximate=True.
- CPI (EUA): sem fonte gratuita automatizável verificada; fica de fora (ver INTEGRATION.md).
"""
from __future__ import annotations

import json
import re
import urllib.request

import pandas as pd

COPOM_API = "https://www.bcb.gov.br/api/servico/sitebcb/copom/comunicados?quantidade={n}"
FOMC_PAGE = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
COLUMNS = ["event", "scheduled_at", "source", "approximate", "detail"]

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ai-trader-research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def copom(get=_get, n: int = 80) -> pd.DataFrame:
    data = json.loads(get(COPOM_API.format(n=n)))["conteudo"]
    rows = [
        {
            "event": "COPOM",
            "scheduled_at": pd.Timestamp(f"{d['dataReferencia']} 18:30").tz_localize("America/Sao_Paulo").tz_convert("UTC"),
            "source": "bcb.gov.br",
            "approximate": False,
            "detail": d["titulo"],
        }
        for d in data
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_fomc(html: str) -> pd.DataFrame:
    rows = []
    for panel in re.split(r"(?=<h4><a id=\"\d+\">\d{4} FOMC Meetings)", html):
        year_m = re.search(r"(\d{4}) FOMC Meetings", panel)
        if not year_m:
            continue
        year = int(year_m.group(1))
        pairs = re.findall(
            r"fomc-meeting__month[^>]*><strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<", panel, re.S
        )
        for month_text, date_text in pairs:
            if "unscheduled" in date_text.lower() or "notation" in date_text.lower():
                continue
            months = [_MONTHS[m.strip()[:3].lower()] for m in month_text.split("/") if m.strip()[:3].lower() in _MONTHS]
            days = re.findall(r"\d+", date_text)
            if not months or not days:
                continue
            month = months[-1]  # "Apr/May 30-1": decisão no último dia, no segundo mês
            rows.append(
                {
                    "event": "FOMC",
                    "scheduled_at": pd.Timestamp(year=year, month=month, day=int(days[-1]), hour=14)
                    .tz_localize("America/New_York")
                    .tz_convert("UTC"),
                    "source": "federalreserve.gov",
                    "approximate": False,
                    "detail": f"{month_text} {date_text}".strip(),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def fomc(get=_get) -> pd.DataFrame:
    return parse_fomc(get(FOMC_PAGE))


def payroll(since: str, until: str) -> pd.DataFrame:
    rows = []
    for month_start in pd.date_range(since, until, freq="MS"):
        first_friday = month_start + pd.offsets.Week(weekday=4) if month_start.weekday() != 4 else month_start
        rows.append(
            {
                "event": "PAYROLL",
                "scheduled_at": (first_friday + pd.Timedelta(hours=8, minutes=30))
                .tz_localize("America/New_York")
                .tz_convert("UTC"),
                "source": "rule:first-friday",
                "approximate": True,
                "detail": "",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


# eventos que importam para cada ativo
RELEVANT = {
    "BTC": {"FOMC", "PAYROLL"},
    "IBOV": {"COPOM", "FOMC", "PAYROLL"},
    "PETR4": {"COPOM", "FOMC"},
    "VALE3": {"COPOM", "FOMC"},
    "USDBRL": {"COPOM", "FOMC", "PAYROLL"},
}
