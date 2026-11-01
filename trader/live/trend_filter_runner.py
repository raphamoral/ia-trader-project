"""Comitê + filtro (meta-labeling) em papel, ao lado do comitê puro (trend_runner)."""
from trader.live.trend_runner import main

if __name__ == "__main__":
    main(filtered=True)
