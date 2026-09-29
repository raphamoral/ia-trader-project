"""Agente "black" de notícias: redes sociais, blogs anônimos e sites de conteúdo em massa.

Fica estritamente separado do agente de notícias confiáveis (`trader.news`):
dados em data/news_black/, colunas de features com prefixo `black_`, e nenhuma
fonte daqui é promovida automaticamente para o conjunto confiável. Essa
separação é o que torna válido o experimento A/B/B'/C em `experiment.py`.
"""
