# Contributing

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv can install it).

```bash
uv sync --frozen
uv run python tasks.py check      # lint, typecheck, test, audit, baseline
```

If the repository sits on a small or slow drive, keep the virtual environment elsewhere:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/<repo>"
uv sync --frozen
```

## Tasks

`tasks.py` is the cross-platform task runner; the Makefile only delegates to it.

| Target | What it runs |
|---|---|
| `lint` | `ruff check` and `ruff format --check` |
| `format` | `ruff format` and `ruff check --fix` |
| `typecheck` | `mypy --strict` over src, tests, scripts, bench and tasks.py |
| `test` | pytest with branch coverage; fails below 85% |
| `bench` | micro-benchmarks with regression thresholds from `bench/thresholds.json` |
| `quality` | recall, NDCG, bias, shilling, canary and the ablations, against their thresholds |
| `load` | the fixed-rate HTTP load test: p50/p99 at 50, 100 and 200 RPS |
| `demo` | the offline end-to-end demo (no API key, no network) |
| `audit` | bandit and pip-audit |
| `baseline` | `scripts/check_baseline.py` |
| `check` | lint, typecheck, test, audit, baseline |
| `full` | `check` plus all three benches (several minutes) |

## Rules for changes

- **Tests first for behavior.** Mark tests `unit` or `integration`. Use hypothesis for properties
  (invariants, round trips, oracles) rather than hand-picked examples alone.
- **Offline by default.** Tests, demo and benches must not touch the network and must not download a
  model or a dataset; `tests/conftest.py` blocks non-loopback sockets. All data comes from the seeded
  generator in `data/generator.py`.
- **Keep the test suite fast.** Tests that need a trained model use the session-scoped
  `tiny_training` fixture rather than training their own, and treat it as read-only — anything that
  mutates an index takes a `copy()` first.
- **Evidence for claims.** A new README claim needs a row in `CLAIMS.md` pointing to a test,
  bench script or CI job. Performance numbers must be copied from real `bench/` output together
  with the command that produced them.
- **Threat model.** New entry points, tools or artifacts need a STRIDE row in
  `docs/THREAT_MODEL.md` with a control and a test.
- **Decisions.** Record significant trade-offs as ADRs in `docs/adr/`.
- **No pickle.** Use JSON, npz with `allow_pickle=False`, LightGBM text or faiss serialization behind
  a hash check. A test bans the pickle family from `src/`.
- **Determinism is a test, not a hope.** Anything that goes into the bundle has to be reproducible:
  `tests/integration/test_pipeline.py` trains twice and compares artifact digests. Watch for
  parallel floating-point reductions (DuckDB aggregates, faiss builds with more than one thread) and
  for sorts whose tie-break depends on them.
- **No secrets in git.** Configuration goes in `.env` (ignored); document new settings in
  `.env.example` (a test enforces that it lists every setting).
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`).
