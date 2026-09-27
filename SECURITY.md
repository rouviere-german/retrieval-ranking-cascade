# Security policy

## Supported versions

Only the latest commit on `main` is supported. This is a portfolio project, not a product with
release branches.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting ("Report a vulnerability" under the repository's Security tab). Include:

- the affected commit and component;
- steps to reproduce, ideally as a failing test;
- the impact you expect.

You can expect an acknowledgement within 7 days and a fix or a written assessment within 30 days.
Credit is given in the fix commit unless you ask otherwise.

## Scope

In scope: the code in this repository, its container image and its CI configuration. Findings about
the serving surface (`/v1/recommend`, `/v1/feedback`, `/v1/admin/*`), the bundle verification path,
the query cost cap and the audit chain are especially welcome.

Out of scope: denial of service by traffic volume, findings that need a compromised host, issues in
third-party dependencies that are already publicly tracked (please report those upstream), and the
limitations this repo already documents and measures — in particular that the tenure-weighted
shilling defence is defeated by accounts with more than a few sessions, that the budget controller
bounds planned work rather than interrupting a running stage, and that the audit log is
tamper-evident rather than tamper-proof. Those are in `docs/THREAT_MODEL.md` under Residual risk and
in the README's Non-goals.

## How this repository is hardened

- Threats, controls and the tests that prove them: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- Every security claim is linked to evidence in [CLAIMS.md](CLAIMS.md).
- CI runs ruff, mypy --strict, pytest with a coverage floor, bandit, semgrep (registry packs plus
  `.semgrep/rules.yml`), pip-audit, gitleaks, a training smoke with metric thresholds, the quality
  and load benches against their thresholds, hadolint, a container build and smoke test, a trivy
  image scan and a syft CycloneDX SBOM.
- Secrets come only from the environment; only `.env.example` is committed, and it holds no values.
- API keys are stored as SHA-256 digests; the process never holds a raw key. Admin operations need a
  second, separately configured scope and are appended to a hash-chained audit log.
- Every model artifact's SHA-256 is verified against the bundle manifest before any parser sees the
  bytes, and nothing is pickled: `.npz` is loaded with `allow_pickle=False` and the ranker is text.
- Synthetic, PII-free data. User ids reach logs and audit records only as keyed, truncated digests.
