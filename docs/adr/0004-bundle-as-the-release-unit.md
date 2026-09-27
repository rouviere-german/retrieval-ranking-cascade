# 4. Release the tower, index, ranker and feature snapshot as one verified bundle

- Status: accepted
- Date: 2026-09-27

## Context

A cascade has four artifacts that have to agree, and nothing about them looks wrong when they do not.

- The item tower produces the embeddings the ANN index is built from. An index built from other
  embeddings still returns *k* neighbours, with plausible scores, from a different space.
- The feature snapshot supplies the towers' side features, so it changes the embeddings. It also
  supplies the ranker's features, so it changes the scores.
- `tower_score` is one of the ranker's ten features. Retraining the tower without retraining the
  ranker silently redefines a feature the ranker learned splits on.
- The drift monitors compare live traffic against the distributions the ranker was *fitted* on, so the
  reference belongs to the same release as the ranker.

Versioning these independently means a deployment can combine four individually valid artifacts into an
invalid system, and every symptom is a quality regression with no stack trace. The usual answer —
four model-registry entries with a naming convention and a wiki page about which versions go together —
puts the consistency guarantee in a human's head.

There is a second, separate problem. Two of the four artifacts are parsed by C++ code: faiss'
`deserialize_index` and LightGBM's `Booster(model_str=...)`. A writable model directory is therefore a
code-execution surface, and "we only load our own files" is not a control.

## Decision

Ship all five artifacts (`tower.npz`, `index.faiss`, `ranker.txt`, `snapshot.npz`, `reference.npz`) as
one directory with a `manifest.json`, addressed by a single `bundle_id`.

**Identity.** `bundle_id` is the SHA-256 of the canonical JSON of the manifest with the id field
removed. The manifest holds a SHA-256 and byte size per artifact, so the id covers the artifact
contents transitively, and a manifest edited to match a modified artifact fails its own id check.

**Load order is the security control.** Read the manifest first; verify its own id; then, for each
artifact, check size and SHA-256 against the manifest; only then let a parser near the bytes. After
deserialization, cross-check what the artifact claims to be against what the manifest says it is: the
index's type, dimension, size and metric; the ranker's feature names against `FEATURE_NAMES`; the
snapshot's content hash; the tower's output dimension. A test flips one byte in each of the five
artifacts in turn and expects five refusals.

**No pickle, anywhere.** `.npz` is loaded with `allow_pickle=False`, the ranker is LightGBM's text
format, the index is faiss' own serialization behind a hash check, and a test bans the pickle family
from `src/`.

**Provenance in the manifest.** The impression-log SHA-256, every seed, the world / tower / index /
ranker / snapshot hyper-parameters, the estimated propensities the IPS weights came from, the
`uv.lock` hash and the installed versions of the numeric dependencies. Enough to answer "what produced
this" without a build system.

**Item embeddings are not stored.** They are recomputed from the item tower and the snapshot at load
time. This costs about a second at startup for 100k items and buys something worth more: the recall
canary compares the index against the vectors *the towers actually produce now*, rather than against a
second stored copy that could drift with it. If the tower and the index disagree, the canary sees it.

**Swapping is an audited admin operation.** `POST /v1/admin/bundle` loads a directory through the same
verification path and swaps it in atomically under a lock, and both success and refusal are appended to
the hash-chained audit log before the response returns.

## Consequences

- A deployment is one identifier. `GET /v1/bundle` returns it along with the live index digest, and
  `cascade_bundle_info` carries it as a metric label, so a quality regression can be attributed to a
  release.
- Rollback is copying a directory back and calling one endpoint. There is no partial rollback, which is
  the point.
- **`bundle_id` is an integrity name, not a content-addressed one.** `created_at` is inside the hashed
  payload, so two runs of the same config produce byte-identical artifacts under two distinct ids. This
  was a deliberate choice after the reproducibility test surfaced it: the alternative (excluding
  `created_at` from the hash) would leave a manifest field the id does not commit to, and a build
  timestamp is exactly the field an operator would want to trust. Reproducibility is therefore asserted
  on the artifact digests and the metric hash, and the test says so.
- Every deployment carries all five artifacts even when only one changed. At `full` scale the index is
  the bulk of it. For a repo that trains in under a minute this is irrelevant; a system retraining a
  ranker hourly against a monthly tower would find it wasteful and should split the bundle — and would
  then need a compatibility matrix, which is the cost this decision avoids.
- Loading is slower than mapping files: five SHA-256 passes plus recomputing 100k item embeddings. It
  happens once at startup, and `/readyz` reports not-ready until the index matches the bundle.
- The verification path is only as good as its coverage, so it is tested exhaustively rather than
  representatively: one byte flipped in each artifact, each artifact deleted, an artifact truncated, a
  manifest that drops an entry and re-derives its id, a manifest that lies about the snapshot hash, a
  manifest that lies about the embedding dimension, an index of the wrong faiss type, an index with the
  wrong metric, and a ranker whose feature names do not match.
- **A verified bundle is not a trusted bundle.** The hashes prove the bytes are the ones the manifest
  describes; they say nothing about whether the model behind them is any good, or about who wrote the
  manifest. There is no signature and no key management here, so an attacker who can write the whole
  directory can write a consistent one. What the control buys is that they cannot corrupt *part* of it,
  and that a swap they perform is on the record.

## Alternatives considered

- **Four independently versioned registry entries with a documented compatibility matrix.** The
  standard approach, and appropriate when components genuinely have separate lifecycles. Rejected here
  because all four are retrained together by one pipeline, so the matrix would have exactly one valid
  row per release and would only ever be a chance to get it wrong.
- **One pickle per release.** Smaller and simpler, and unacceptable: loading it is arbitrary code
  execution, and the portfolio baseline bans it outright.
- **Storing item embeddings in the bundle** instead of recomputing them. Faster startup, and it would
  turn the recall canary into a comparison between two stored copies of the same thing — which is a
  weaker check precisely where a silent failure is most likely.
- **Signing the manifest.** The right next step, and deliberately not taken: key distribution, rotation
  and revocation are a system this repo does not have, and a signature with no key management is
  decoration. Recorded as a non-goal instead.
