# Threat model

Scope: the `retrieval-ranking-cascade` service (`POST /v1/recommend`, `POST /v1/feedback`,
`GET /v1/bundle`, `GET /v1/monitoring`, `POST /v1/admin/*`), the bundle it loads from disk, the
training pipeline that produces that bundle, and the offline evaluation that grades it.

Out of scope: the network in front of the service (TLS termination, WAF, DDoS absorption), the
identity provider that issues API keys, the CI/CD system, and the host's own hardening.

## Assets

| Asset | Why it matters |
|---|---|
| Serving bundle (tower, HNSW index, LambdaMART model, feature snapshot, drift reference) | Determines every recommendation. A swapped or corrupted artifact changes behaviour silently. |
| Item statistics inside the feature snapshot | The most attacker-reachable input: anyone who can generate clicks can move them. |
| Latency budget | The whole service's contract with its caller. Exhausting it is the cheapest denial of service. |
| Admin audit log | The only record of who reindexed or swapped a bundle. |
| API keys, admin scope | Access to serving and to privileged operations. |
| Impression log and user ids | Synthetic here, but the code paths are the ones real logs would flow through. |

## Trust boundaries

1. **Untrusted caller -> HTTP surface.** Anything in a request body or header is attacker-controlled,
   including a valid API key's owner asking for expensive work.
2. **Untrusted feedback -> item statistics.** Feedback events look like legitimate traffic by
   construction; no input validation can tell a real click from a paid one.
3. **Disk -> process.** Bundle bytes are parsed by C++ readers (faiss, LightGBM). A writable bundle
   directory is a code-execution surface unless the bytes are verified first.
4. **Admin caller -> bundle and index.** Reindex and bundle swap change what every subsequent
   request returns.
5. **Training data -> model.** The impression log decides what the model learns; a contract
   violation that trains silently is worse than a crash.

## STRIDE

### Spoofing

| Threat | Control | Test |
|---|---|---|
| Unauthenticated caller reaches a serving or admin route | API key required on every route except `/healthz` and `/readyz`; keys held as SHA-256 digests and compared in constant time; protected routes fail closed when no keys are configured | `tests/integration/test_cascade_api.py::test_every_post_route_requires_a_key`, `tests/integration/test_cascade_api.py::test_every_get_route_requires_a_key`, `tests/unit/test_security.py::test_no_configured_keys_fails_closed` |
| Forged or guessed key | Digest comparison is constant time; a key is 32 bytes of `secrets` entropy | `tests/unit/test_security.py::test_generated_keys_verify_against_their_digest`, `tests/unit/test_security.py::test_every_configured_digest_is_compared_without_early_exit`, `tests/integration/test_cascade_api.py::test_an_unknown_key_is_rejected` |
| Feedback attributed to a source that did not send it | `source` is a bounded 12-hex identifier and only selects a rate-limit bucket; it is never trusted as an identity and never reaches training | `tests/integration/test_cascade_api.py::test_the_feedback_schema_rejects_malformed_bodies`, `tests/unit/test_monitoring.py::test_feedback_updates_the_simulated_ctr_gauge_but_never_training` |

### Tampering

| Threat | Control | Test |
|---|---|---|
| A bundle artifact is modified on disk | Every artifact's SHA-256 is compared with the manifest before any parser sees the bytes; size is checked too | `tests/unit/test_bundle.py::test_flipping_one_byte_in_any_artifact_is_caught_before_it_is_parsed`, `tests/unit/test_bundle.py::test_truncating_an_artifact_is_caught_by_its_recorded_size` |
| The manifest itself is edited to match a modified artifact | `bundle_id` is the SHA-256 of the manifest with the id removed, so an edited manifest fails its own id check | `tests/unit/test_bundle.py::test_editing_the_manifest_breaks_its_own_id`, `tests/unit/test_bundle.py::test_the_id_covers_the_artifact_digests_transitively` |
| A manifest drops an artifact, or lies about the snapshot hash or the embedding dimension, and re-derives its id | Load requires every artifact by name and cross-checks the snapshot's content hash and the tower's output dimension against the manifest | `tests/unit/test_bundle.py::test_a_manifest_that_omits_a_required_artifact_is_refused`, `tests/unit/test_bundle.py::test_a_manifest_that_lies_about_the_snapshot_hash_is_refused`, `tests/unit/test_bundle.py::test_a_manifest_that_lies_about_the_embedding_dimension_is_refused` |
| Malicious bytes reach the faiss or LightGBM C++ parser | Hash check first; then type, dimension, size and metric are verified on the deserialized index, and the ranker's feature names are checked against `FEATURE_NAMES` | `tests/unit/test_ann.py::test_a_tampered_index_never_reaches_the_faiss_parser`, `tests/unit/test_ann.py::test_an_index_of_the_wrong_type_or_metric_is_refused`, `tests/unit/test_ranker.py::test_a_tampered_model_text_never_reaches_the_lightgbm_parser`, `tests/unit/test_ranker.py::test_a_model_whose_features_do_not_match_is_refused` |
| Arbitrary code via pickle in a model or array file | No pickle anywhere: `.npz` is loaded with `allow_pickle=False`, the ranker is text, a lint check bans the pickle family from `src/` | `tests/unit/test_no_pickle.py::test_source_tree_has_no_pickle_family_serialization`, `tests/unit/test_ml_serialization.py::test_pickled_npz_files_are_refused_on_load`, `tests/unit/test_bundle.py::test_nothing_in_a_bundle_is_pickled` |
| A poisoned training log trains silently | pandera contracts on the impression log and the catalog before training; the feature snapshot re-validates its own shapes and ranges on every load | `tests/unit/test_contracts.py::test_out_of_range_columns_are_rejected`, `tests/unit/test_features.py::test_the_snapshot_contract_rejects_broken_arrays`, `tests/unit/test_generator.py::test_log_and_catalog_satisfy_their_contracts` |
| Fake accounts push a chosen item up the ranking | Item statistics count each user at most once per item, weight each user by tenure and shrink item CTR towards the global rate; the naive aggregation is measured alongside as a control | `tests/integration/test_evaluation.py::test_robust_aggregation_damps_a_shilling_attack_the_naive_one_falls_for`, `tests/unit/test_features.py::test_robust_item_counts_weight_a_fresh_account_below_an_established_one`, `tests/unit/test_features.py::test_a_repeated_click_on_the_same_item_counts_once_under_robust_aggregation`, `bench/bench_quality.py` |
| Live feedback silently retrains the model | Accepted feedback only moves gauges; it never touches the snapshot or the model | `tests/unit/test_monitoring.py::test_feedback_updates_the_simulated_ctr_gauge_but_never_training` |

**Residual risk.** Tenure weighting saturates at the trust threshold. The bench measures 200
accounts with four sessions each defeating it (target rank 46 of 10000, damping 0.58x), against 200
one-session accounts being damped 27x. Account-age gates, per-account rate limits and graph-based
collusion detection are out of scope; the bench reports the failure rather than hiding it.

### Repudiation

| Threat | Control | Test |
|---|---|---|
| An operator denies reindexing or swapping a bundle | Every privileged action is appended to a hash-chained JSONL log with the actor's key id, under a file lock and fsynced, before the response returns | `tests/integration/test_cascade_api.py::test_reindex_swaps_the_index_and_records_the_new_digest_in_the_chain`, `tests/integration/test_cascade_api.py::test_swapping_in_a_verified_bundle_is_accepted_and_audited` |
| A refused privileged action leaves no trace | A rejected bundle swap is audited with the exception type before the 422 is returned | `tests/integration/test_cascade_api.py::test_swapping_in_a_tampered_bundle_is_refused_and_the_refusal_is_audited`, `tests/integration/test_cascade_api.py::test_swapping_from_a_directory_that_does_not_exist_is_refused_and_audited` |
| An audit entry is edited, deleted or reordered | Each entry's hash covers its own payload including `prev`; verification localizes the break | `tests/unit/test_audit.py::test_editing_an_entry_without_rehashing_breaks_its_own_hash`, `tests/unit/test_audit.py::test_rehashing_an_edited_entry_still_breaks_the_link_to_the_next_one`, `tests/unit/test_audit.py::test_deleting_an_entry_is_caught_as_a_sequence_gap`, `tests/unit/test_audit.py::test_reordering_two_entries_is_caught`, `tests/unit/test_audit.py::test_flipping_any_single_entry_is_always_detected` |
| The head hash cannot be pinned externally | `GET /v1/bundle` exposes `audit_head`, and the CLI can assert it | `tests/integration/test_cascade_api.py::test_the_admin_audit_chain_verifies_after_a_run_of_actions`, `tests/integration/test_demos_cli.py::test_audit_verify_accepts_a_good_chain_and_can_pin_the_head` |

**Residual risk.** The log is tamper-*evident*, not tamper-proof. Whoever can write the file can
truncate its tail; detecting that after the fact needs the head hash stored elsewhere.

### Information disclosure

| Threat | Control | Test |
|---|---|---|
| An error leaks a stack trace, an exception message or a submitted value | Fixed error codes and messages; unhandled exceptions become a generic 500 | `tests/integration/test_app.py::test_unhandled_exceptions_become_generic_500s`, `tests/integration/test_app.py::test_validation_errors_never_echo_submitted_values` |
| User ids appear in logs or in the audit trail | Ids are replaced by keyed, truncated HMAC digests; the key is random per process, so a small id space cannot be enumerated | `tests/unit/test_audit.py::test_the_digest_is_keyed_so_a_small_id_space_cannot_be_enumerated`, `tests/unit/test_audit.py::test_the_pseudonymizer_is_stable_for_a_key_and_distinct_across_users`, `tests/integration/test_cascade_api.py::test_user_ids_never_reach_the_logs_in_the_clear` |
| A secret handed to the audit log is written verbatim | Audit details pass through the redaction layer before they are written | `tests/unit/test_audit.py::test_secrets_in_details_are_redacted_before_they_are_written`, `tests/unit/test_redaction.py::test_redact_value_replaces_sensitive_keys_and_scrubs_nested_text` |
| Metrics or the monitoring route are readable without a key | `/metrics` requires a key when `APP_METRICS_REQUIRE_AUTH` is set (the default); `/v1/monitoring` always does | `tests/integration/test_cascade_api.py::test_every_get_route_requires_a_key`, `tests/unit/test_config.py::test_defaults_fail_closed` |
| Cross-origin reads of an authenticated response | CORS admits only explicitly configured origins; `*` is refused at config time | `tests/unit/test_config.py::test_wildcard_or_malformed_origins_are_rejected` |

### Denial of service

| Threat | Control | Test |
|---|---|---|
| A caller asks for a very wide ANN search | `candidate_k` and `ef_search` are individually bounded, and a query cost cap rejects expensive combinations of individually legal values | `tests/integration/test_cascade_api.py::test_out_of_bounds_parameters_are_refused_with_a_specific_code`, `tests/integration/test_cascade_api.py::test_the_cost_cap_refuses_a_pair_that_is_individually_legal`, `tests/integration/test_cascade_api.py::test_the_query_cost_grows_with_both_knobs_and_validates_them` |
| Request volume exhausts the process | Per-client token bucket with a bounded key table, and a body-size limit | `tests/unit/test_ratelimit.py::test_burst_is_admitted_then_refills_at_rate`, `tests/unit/test_ratelimit.py::test_tracked_keys_are_bounded`, `tests/integration/test_app.py::test_body_size_limit_applies_to_declared_and_streamed_bodies` |
| Load pushes p99 past the budget | Per-stage budgets with a three-step degradation ladder driven by measured cost, queue time and in-flight contention; responses that lost quality are flagged and counted | `tests/unit/test_budget.py::test_a_tight_retrieval_budget_lowers_ef_search_first`, `tests/unit/test_budget.py::test_a_tight_ranking_budget_lowers_the_rerank_depth`, `tests/unit/test_budget.py::test_a_budget_that_cannot_fit_the_minimum_depth_skips_reranking_altogether`, `tests/unit/test_cascade.py::test_a_budget_too_small_for_the_reranker_returns_ann_order_flagged_degraded`, `bench/bench_load.py` |
| The degradation ladder loops or degrades without bound | The ladder terminates at documented floors (`efSearch` 16, candidates 16, depth 32) and a plan never grows the request | `tests/unit/test_budget.py::test_the_ladder_stops_at_the_floors_even_when_nothing_fits`, `tests/unit/test_budget.py::test_a_retrieval_plan_never_grows_the_request_and_respects_the_floors` |
| One feedback source floods the statistics | Per-source token bucket with `Retry-After`, independent of other sources | `tests/integration/test_cascade_api.py::test_one_noisy_source_cannot_flood_the_statistics`, `tests/integration/test_cascade_api.py::test_a_rate_limited_source_is_told_when_to_retry` |
| A monitoring buffer grows with traffic | Ring buffers with a fixed capacity, and feature rows recomputed only for a sampled subset of responses | `tests/unit/test_monitoring.py::test_the_feature_buffer_is_bounded`, `tests/unit/test_monitoring.py::test_feature_rows_are_only_recomputed_for_a_sampled_subset` |

**Residual risk.** The controller bounds *planned* work only; a stage that has started is never
interrupted, so one pathological request can overshoot. Buckets and estimators are per process, so
several replicas budget independently.

### Elevation of privilege

| Threat | Control | Test |
|---|---|---|
| A serving key performs a privileged operation | `/v1/admin/*` requires the key digest to be listed in `APP_ADMIN_API_KEY_SHA256` as well | `tests/integration/test_cascade_api.py::test_a_serving_key_cannot_reach_an_admin_route` |
| An admin swaps in a bundle from an arbitrary directory | The bundle is fully hash-verified before it is loaded; a failure leaves the live bundle untouched and is audited | `tests/integration/test_cascade_api.py::test_swapping_in_a_tampered_bundle_is_refused_and_the_refusal_is_audited` |
| A reindex quietly invalidates the bundle's own hashes | The bundle on disk is never rewritten; the new index digest is audited and exposed by `GET /v1/bundle` | `tests/integration/test_cascade_api.py::test_reindex_swaps_the_index_and_records_the_new_digest_in_the_chain` |
| Reindex parameters are used to exhaust memory or CPU | `m`, `ef_construction` and `ef_search` are bounded by the request schema and again by `HnswParams` | `tests/integration/test_cascade_api.py::test_reindex_parameters_are_bounded`, `tests/unit/test_ann.py::test_hnsw_params_are_bounded` |
| Secrets come from a file in the image | Configuration is environment-only; only `.env.example` is committed and it holds no values | `tests/unit/test_config.py::test_env_example_documents_every_setting`, `scripts/check_baseline.py` |

## Silent-failure threats specific to this system

These are not STRIDE categories — nobody is attacking — but they are the failures that actually
happen to a cascade, and each one has a detector.

| Failure | Detector | Test |
|---|---|---|
| ANN recall decays after a rebuild or a partial overwrite; the index still returns plausible neighbours | Recall canary comparing the live index with exact search over sampled real user vectors, alerting below a threshold | `tests/unit/test_canary.py::test_a_corrupted_index_alerts_within_one_cycle`, `tests/unit/test_monitoring.py::test_the_canary_alerts_and_counts_after_the_live_index_is_corrupted`, `tests/integration/test_cascade_api.py::test_a_corrupted_live_index_makes_the_canary_alert_on_the_next_cycle` |
| The served feature distribution drifts away from the one the ranker was fitted on | Per-feature PSI against the bundle's stored training reference | `tests/unit/test_monitoring.py::test_psi_is_reported_per_feature_once_the_buffer_is_full_enough`, `tests/unit/test_monitoring.py::test_served_traffic_drifts_away_from_the_training_reference` |
| A bundle swap changes the item tower while the unit-norm outputs still look fine | PSI of the pre-normalization embedding norms against the reference | `tests/unit/test_monitoring.py::test_a_changed_item_tower_shows_up_as_embedding_norm_drift` |
| A hand-derived gradient is wrong, so the tower trains to something plausible but incorrect | Every analytic gradient entry compared with a central finite difference, over a range of shapes and both logQ settings | `tests/unit/test_two_tower.py::test_analytic_gradients_match_central_differences` |
| Training stops being reproducible, so a metric change cannot be attributed | Two full training runs compared on artifact digests and a metric hash | `tests/integration/test_pipeline.py::test_the_same_config_reproduces_the_data_hash_and_the_metric_hash`, `tests/unit/test_features.py::test_the_same_log_always_produces_the_same_snapshot_bytes` |
| Training features are computed from the labels they predict | The training snapshot aggregates the history period only; the serving snapshot adds the label period | `tests/integration/test_pipeline.py::test_training_features_never_see_the_label_period` |
| An index is built from embeddings its consumer does not recompute, making recall meaningless | Item vectors are recomputed from the tower at load time rather than stored | `tests/unit/test_bundle.py::test_item_vectors_are_recomputed_from_the_tower_not_stored`, `tests/integration/test_pipeline.py::test_the_serving_index_matches_the_serving_snapshot_s_vectors` |
| Offline click metrics mislead a model decision | IPS-corrected metrics reported next to naive and ground-truth ones, with the bias in both directions measured | `tests/integration/test_evaluation.py::test_ips_shrinks_the_gap_between_estimated_and_true_per_rank_relevance`, `tests/integration/test_evaluation.py::test_the_ips_advantage_is_closer_to_the_truth_than_the_naive_one`, `bench/bench_quality.py` |
| The index is loaded but does not match the bundle, and traffic is routed anyway | `/readyz` fails while the index size differs from the bundle's item count | `tests/integration/test_cascade_api.py::test_readyz_fails_while_the_index_does_not_match_the_bundle` |

## Continuous verification

`.github/workflows/ci.yml` runs, on every push and pull request:

- `ci.yml#lint` (ruff lint and format), `ci.yml#typecheck` (mypy `--strict`)
- `ci.yml#test` (pytest, 85% coverage floor, hypothesis `ci` profile)
- `ci.yml#bandit`, `ci.yml#semgrep` (registry plus `.semgrep/rules.yml`), `ci.yml#pip-audit`,
  `ci.yml#gitleaks`
- `ci.yml#baseline` (`scripts/check_baseline.py`, which also checks that every evidence reference in
  this file resolves to a real test)
- `ci.yml#bench` (`bench/bench_baseline.py --check`), `ci.yml#train-smoke` (a 1% training run with
  metric thresholds), `ci.yml#quality` and `ci.yml#load` (the quality and latency thresholds)
- `ci.yml#docker` (hadolint, build, container smoke test asserting a non-root uid, trivy image scan,
  syft CycloneDX SBOM)

## Runbook: drift and recall alerts

**`cascade_ann_recall_canary` below threshold / `cascade_ann_recall_canary_alerts_total` rising.**
The index no longer agrees with the item vectors the towers produce. Check `GET /v1/bundle` for
`live_index_sha256` against the manifest's `index.faiss` digest — if they differ, a reindex is live
and its parameters are in the audit log. `POST /v1/admin/reindex` rebuilds from the bundle's own item
vectors and is the first thing to try; the demo shows recall recovering after one. If a reindex does
not fix it, the item vectors themselves are wrong, which means the tower or the snapshot is wrong:
verify the bundle with `retrieval-ranking-cascade bundle <dir>` and roll back to the previous bundle
directory with `POST /v1/admin/bundle`.

**`cascade_feature_psi{feature=...}` above 0.2.** The served feature distribution has moved away from
the training reference. Two benign causes come first: the serving snapshot is fresher than the
training snapshot by construction (one label period), and served rows are the cascade's own top
picks rather than a random slice. Compare against the same feature's PSI right after the bundle was
deployed, which `GET /v1/monitoring` reports from an empty buffer as it fills. A step change that
coincides with a bundle swap points at the new bundle; a slow rise points at a stale snapshot and a
retrain.

**`cascade_embedding_norm_drift` above 0.2.** The item tower's pre-normalization norms have moved,
which means the model changed even if its unit-norm outputs look similar. Cross-check the bundle id
in `cascade_bundle_info` against what was intended to be deployed.

**`cascade_degraded_responses_total` rising.** Read the `reason` label. `lower_ef_search` means stage
1 is over its allowance — usually contention, sometimes a too-wide `candidate_k` from a caller.
`lower_rerank_depth` and `skip_rerank` mean stage 2 is. Compare
`cascade_stage_latency_seconds{stage="queue"}` with the stage histograms: a large queue component
with normal stage times is a capacity problem, not a cascade problem. Degradation with no queue time
and no contention means a cost estimator has drifted; restarting resets it to the documented
defaults.

**`cascade_catalog_coverage` collapsing.** The service is serving the same items to everyone. Check
the canary first (a broken index concentrates results), then `cascade_feature_psi` on
`item_log_clickers` and `item_ctr` for a feedback-poisoning pattern, then
`cascade_feedback_rejected_total{reason="rate_limited"}` for a source that has been pushing hard.
