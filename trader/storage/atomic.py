"""Troca atômica de arquivo que tolera leitores concorrentes no Windows.

No Windows, os.replace falha com PermissionError se outro processo estiver com
o arquivo de destino aberto (ex.: um fork lendo notícias enquanto o download
grava). Em vez de derrubar o processo, espera o leitor soltar e tenta de novo.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


def replace_with_retry(src: str | Path, dst: str | Path, tries: int = 40, delay_s: float = 0.25) -> None:
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(delay_s * min(attempt + 1, 8))  # até ~1 min no total


def unique_tmp(path: str | Path) -> Path:
    """Temporário exclusivo deste processo: dois processos gravando o mesmo arquivo não se atropelam."""
    import uuid

    path = Path(path)
    return path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
