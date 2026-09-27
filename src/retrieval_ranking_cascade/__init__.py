"""Two-tower retrieval, HNSW recall and LambdaMART reranking as a latency-budgeted cascade.

Sub-packages, in the order a request meets them:

- ``core``: settings, logging with PII redaction, tracing, metrics, hardened FastAPI app factory.
- ``data``: the seeded synthetic world, its logging policy and its position-based click model.
- ``ml``: pandera data contracts, run manifests, a content-addressed registry, PSI/KS drift
  monitors and pickle-free serialization.
- ``retrieval``: the NumPy two-tower model with hand-derived gradients, the faiss HNSW index and
  the recall canary that watches it.
- ``ranking``: the DuckDB feature snapshot, the LightGBM lambdarank reranker and the business
  rules (seen filter, MMR diversity, category caps).
- ``serving``: the cascade itself, the per-stage budget controller, the verified bundle registry,
  the HTTP surface, the live monitors and the hash-chained admin audit log.
- ``evaluation``: ranking metrics, propensity estimation from result randomization, the two offline
  protocols, the shilling-attack harness and the feedback simulator.

``pipeline.py`` wires the offline half together; ``demo.py`` runs the whole thing offline and then
breaks it on purpose.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
