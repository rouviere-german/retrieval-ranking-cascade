# Claims and evidence

Every claim README.md makes, with the test, benchmark or CI job that backs it. A claim with no
evidence does not belong in the README, and `scripts/check_baseline.py` fails if any reference here
points at something that does not exist.

Numbers are from the run recorded in README.md's Results section: Windows 10, 6 logical CPUs,
CPU-only, Python 3.12.14, 2026-09-27, `small` scale.

## The primary pattern: a latency-budgeted cascade

| Claim | Evidence |
|---|---|
| Three stages run in order: ANN retrieval, LambdaMART reranking, business rules | `tests/unit/test_cascade.py::test_a_known_user_gets_k_ranked_items_from_every_stage` |
| Every stage is timed and the timings are returned to the caller | `tests/unit/test_cascade.py::test_a_known_user_gets_k_ranked_items_from_every_stage`, `tests/integration/test_cascade_api.py::test_a_recommendation_carries_its_provenance_and_stage_timings` |
| The seen filter runs between stages 1 and 2, so the expensive stage never scores a dropped item | `tests/unit/test_cascade.py::test_items_the_user_already_clicked_are_never_recommended`, `tests/unit/test_rules.py::test_allowed_positions_preserves_the_ann_order` |
| Stage cost is predicted from measured cost as `fixed + rate * units` | `tests/unit/test_budget.py::test_the_estimator_predicts_fixed_plus_rate_times_units_with_a_safety_margin`, `tests/unit/test_budget.py::test_the_estimator_converges_on_a_stable_cost` |
| What the cost model says fits, actually fits (the invariant the ladder rests on) | `tests/unit/test_budget.py::test_max_units_never_predicts_over_the_allowance`, `tests/unit/test_budget.py::test_a_ranking_plan_is_a_prefix_and_its_prediction_fits_when_it_reranks` |
| Step 1 of the ladder lowers `efSearch` and keeps reranking | `tests/unit/test_budget.py::test_a_tight_retrieval_budget_lowers_ef_search_first`, `tests/unit/test_cascade.py::test_a_tight_retrieval_budget_lowers_ef_search_but_keeps_reranking` |
| Step 2 lowers the rerank depth | `tests/unit/test_budget.py::test_a_tight_ranking_budget_lowers_the_rerank_depth` |
| Step 3 skips reranking and returns ANN order flagged `degraded` | `tests/unit/test_budget.py::test_a_budget_that_cannot_fit_the_minimum_depth_skips_reranking_altogether`, `tests/unit/test_cascade.py::test_a_budget_too_small_for_the_reranker_returns_ann_order_flagged_degraded`, `tests/unit/test_cascade.py::test_the_degraded_order_is_the_ann_order_and_differs_from_the_reranked_one` |
| A degraded response is still a real recommendation list, not an error | `tests/unit/test_cascade.py::test_a_budget_too_small_for_the_reranker_returns_ann_order_flagged_degraded`, `tests/integration/test_cascade_api.py::test_a_degraded_response_is_labelled_and_counted` |
| The ladder terminates at documented floors and never grows the request | `tests/unit/test_budget.py::test_the_ladder_stops_at_the_floors_even_when_nothing_fits`, `tests/unit/test_budget.py::test_a_retrieval_plan_never_grows_the_request_and_respects_the_floors` |
| Queue time is already spent when the budget starts | `tests/unit/test_budget.py::test_queue_time_is_already_spent_when_the_budget_starts`, `tests/unit/test_budget.py::test_time_already_spent_shrinks_the_ranking_allowance` |
| In-flight contention scales the predictions, and can force degradation on its own | `tests/unit/test_budget.py::test_contention_makes_the_expensive_stage_look_as_expensive_as_it_is`, `tests/unit/test_budget.py::test_contention_scales_the_retrieval_prediction_too`, `tests/unit/test_cascade.py::test_contention_alone_can_force_degradation` |
| An unknown user is served from the popularity list and flagged `cold_start` | `tests/unit/test_cascade.py::test_an_unknown_user_is_served_from_the_popularity_list_and_flagged`, `tests/integration/test_cascade_api.py::test_an_unknown_user_is_answered_from_popularity_and_flagged` |
| A known user with no clicks is not cold start; the tower answers content-only | `tests/unit/test_cascade.py::test_a_known_user_with_no_click_history_still_goes_through_the_towers` |
| The cascade is safe to call from several threads and gives the same answers as sequential calls | `tests/unit/test_cascade.py::test_concurrent_requests_all_return_complete_results`, `tests/unit/test_ranker.py::test_each_thread_gets_its_own_parsed_booster` |
| p99 is inside the 50 ms budget at 50 RPS with zero errors (20.5 ms measured; 18.1-26.8 ms across three clean runs) | `bench/bench_load.py`, `.github/workflows/ci.yml#load` |
| Whenever the *server* falls behind the offered rate, at least 10% of responses are degraded | `bench/bench_load.py` (the `keeps_up_rps_fraction` / `min_degraded_rate_when_behind` thresholds, skipped for a harness-limited phase) |
| A phase whose latency came from the load generator rather than the server is detected and reported as harness-limited instead of asserted on | `bench/bench_load.py` (`PhaseResult.harness_limited`, which compares client-observed p50 *and* p99 with the server's own view of the same requests) |

## Stage 1: two-tower retrieval and HNSW

| Claim | Evidence |
|---|---|
| The sampled-softmax gradients are hand-derived and correct: every analytic entry matches a central finite difference | `tests/unit/test_two_tower.py::test_analytic_gradients_match_central_differences` |
| The embedding map is smooth even when every ReLU unit is dead (no NaN, no 1e12 gradients) | `tests/unit/test_two_tower.py::test_a_dead_relu_layer_gives_zero_embeddings_and_finite_gradients`, `tests/unit/test_two_tower.py::test_tower_output_is_unit_norm_where_the_layer_is_alive` |
| Duplicate positives are masked, so a repeated positive is not a negative for its own row | `tests/unit/test_two_tower.py::test_sampled_softmax_masks_duplicate_positives` |
| logQ correction is a no-op when candidates are sampled uniformly and bites when they are not | `tests/unit/test_two_tower.py::test_a_uniform_logq_is_a_no_op_but_a_varying_one_is_not` |
| The lazy (row-wise) Adam update equals the dense one on touched rows and leaves others alone | `tests/unit/test_two_tower.py::test_lazy_adam_matches_a_dense_update_on_touched_rows` |
| Training lowers the loss and is deterministic in the config | `tests/unit/test_two_tower.py::test_training_reduces_the_loss_and_is_deterministic`, `tests/integration/test_pipeline.py::test_the_tower_loss_falls_over_the_epochs` |
| recall@500 against exact search is 0.977 at the default `efSearch` of 768 | `bench/bench_quality.py` (threshold `min_recall_at_default` 0.95), `.github/workflows/ci.yml#quality` |
| Recall rises monotonically with `efSearch`, so it is a real knob | `tests/unit/test_ann.py::test_recall_increases_with_ef_search`, `bench/bench_quality.py` |
| HNSW search agrees with exact search at a generous `efSearch` | `tests/unit/test_ann.py::test_hnsw_search_agrees_with_exact_search_at_a_generous_ef` |
| The exact reference is correct: it matches a full argsort and breaks ties by id | `tests/unit/test_ann.py::test_exact_top_k_matches_a_full_argsort`, `tests/unit/test_ann.py::test_exact_top_k_breaks_ties_towards_the_lower_id` |
| Index builds are reproducible byte for byte with one thread | `tests/unit/test_ann.py::test_building_twice_with_one_thread_gives_the_same_bytes` |
| At 10k items the ANN index is *slower* than brute force (0.3x), reported rather than hidden | `bench/bench_quality.py` (the retrieval block prints the exact baseline next to the ANN curve) |

## Stage 2: LambdaMART reranking

| Claim | Evidence |
|---|---|
| Training queries are label-period sessions with both a clicked and an unclicked item | `tests/unit/test_ranker.py::test_every_query_has_both_a_clicked_and_an_unclicked_row`, `tests/unit/test_ranker.py::test_a_label_period_with_no_usable_session_is_refused` |
| Extra negatives come from the retrieved pool and never include an item the user clicked | `tests/unit/test_ranker.py::test_extra_negatives_are_unshown_rows_at_the_slots_bucket`, `tests/unit/test_ranker.py::test_a_session_never_proposes_an_item_the_user_already_clicked` |
| Shown rows carry clipped inverse-propensity weights; unshown ones carry 1 | `tests/unit/test_ranker.py::test_shown_rows_get_clipped_inverse_propensities_and_unshown_rows_get_one`, `tests/unit/test_ranker.py::test_every_row_weight_is_between_one_and_max_weight` |
| IPS weighting is required when asked for, and refuses to run without a propensity model | `tests/unit/test_ranker.py::test_building_a_dataset_without_a_propensity_model_is_refused` |
| IPS weighting beats naive clicks on corpus NDCG (0.6100 vs 0.5690) and LightGBM's unbiased lambdarank loses it badly (0.4834) | `bench/bench_quality.py --ablations` (the `ranker_labels` table) |
| Training and serving features come from the same function, so they cannot drift apart | `tests/unit/test_features.py::test_a_scalar_user_and_a_user_array_give_the_same_rows` |
| Training features never see the label period | `tests/integration/test_pipeline.py::test_training_features_never_see_the_label_period` |
| Predictions are deterministic and finite | `tests/unit/test_ranker.py::test_predictions_are_deterministic`, `tests/unit/test_ranker.py::test_predictions_are_finite_one_per_row_and_order_dependent_on_features` |
| Each serving thread gets its own parsed booster, which is worth 4.9x the throughput of one booster behind a mutex (2363 against 481 predictions/s at eight callers) | `tests/unit/test_ranker.py::test_each_thread_gets_its_own_parsed_booster`, `bench/bench_baseline.py::bench_ranker_predict_threads` |
| Quality peaks at rerank depth 100 while cost is linear in depth, which is why 100 is the served default | `bench/bench_quality.py --ablations` (the `rerank_depth` table), `tests/unit/test_config.py::test_serving_defaults_match_the_depth_the_ablation_picked` |

## Stage 3: diversity and business rules

| Claim | Evidence |
|---|---|
| At `lambda = 1` MMR is pure relevance order | `tests/unit/test_rules.py::test_mmr_at_lambda_one_is_pure_relevance_order` |
| The first pick is always the most relevant candidate, even at `lambda = 0` | `tests/unit/test_rules.py::test_mmr_at_lambda_zero_still_starts_from_the_most_relevant_item` |
| Lowering `lambda` trades relevance for diversity, with a switch point that follows the formula | `tests/unit/test_rules.py::test_lambda_trades_a_near_duplicate_for_a_dissimilar_item`, `tests/unit/test_cascade.py::test_lambda_one_gives_pure_relevance_order_and_lower_lambda_diversifies`, `bench/bench_quality.py --ablations` (the `mmr_lambda` table) |
| Each greedy pick maximizes the marginal MMR objective (the incremental form equals the formula) | `tests/unit/test_rules.py::test_each_mmr_pick_maximizes_the_marginal_objective` |
| MMR selection is deterministic and returns a valid index set | `tests/unit/test_rules.py::test_mmr_selection_is_deterministic_and_a_valid_index_set`, `tests/unit/test_rules.py::test_mmr_returns_a_permutation_prefix_without_repeats` |
| The category cap is a hard constraint, applied as an order-preserving filter | `tests/unit/test_rules.py::test_cap_per_group_never_exceeds_the_cap_or_k`, `tests/unit/test_rules.py::test_cap_per_group_keeps_the_earliest_items_of_each_group`, `tests/unit/test_cascade.py::test_the_category_cap_is_enforced` |
| A hard cap can return a short page rather than break the constraint | `tests/unit/test_cascade.py::test_the_category_cap_can_return_a_short_page_rather_than_break_the_constraint` |
| MMR costs 0.0041 NDCG and buys lower intra-list similarity plus more cold items | `bench/bench_quality.py` (the corpus block's `diversity_cost`), `tests/integration/test_evaluation.py::test_the_diversity_rules_cost_relevance_and_buy_coverage` |

## Evaluation and bias correction

| Claim | Evidence |
|---|---|
| Propensities are estimated from the randomized slice only | `tests/unit/test_propensity.py::test_only_the_randomized_slice_is_used`, `tests/unit/test_propensity.py::test_estimated_propensities_track_the_true_examination_curve` |
| Estimation fails loudly when rank 1 cannot be identified, rather than papering over it | `tests/unit/test_propensity.py::test_estimation_fails_loudly_when_rank_one_cannot_be_identified` |
| Ranks with too little randomized traffic inherit the nearest reliable rank | `tests/unit/test_propensity.py::test_ranks_with_too_few_randomized_impressions_inherit_the_nearest_reliable_rank` |
| The examination exponent is recovered from the log alone (1.024 against a true 1.0) | `tests/unit/test_propensity.py::test_estimate_eta_recovers_an_exact_power_law`, `bench/bench_quality.py` (threshold `max_eta_error` 0.2), `tests/integration/test_demos_cli.py::test_the_demo_recovers_the_examination_exponent_from_the_randomized_slice` |
| IPS recovers the relevance level a naive CTR understates | `tests/unit/test_propensity.py::test_ips_recovers_the_relevance_level_that_the_naive_ctr_understates` |
| Per-rank bias falls from 0.2513 to 0.0373 with IPS (85.2%) | `tests/integration/test_evaluation.py::test_ips_shrinks_the_gap_between_estimated_and_true_per_rank_relevance`, `bench/bench_quality.py` (threshold `min_position_bias_reduction` 0.5) |
| Policy-advantage bias falls from 0.1406 to 0.1063 (24.4%), a smaller win, reported as such | `tests/integration/test_evaluation.py::test_the_ips_advantage_is_closer_to_the_truth_than_the_naive_one`, `bench/bench_quality.py` (threshold `min_advantage_bias_reduction` 0.15) |
| SNIPS is IPS divided by the mean weight, and stays a probability | `tests/unit/test_propensity.py::test_snips_is_ips_divided_by_the_mean_weight`, `tests/unit/test_propensity.py::test_snips_stays_a_probability_because_it_is_a_convex_combination` |
| The randomized slice is excluded from the bias measurement by default | `tests/integration/test_evaluation.py::test_the_randomized_slice_is_excluded_by_default` |
| The cascade beats popularity and the tower on the slate protocol (NDCG truth 0.8545 / 0.7442 / 0.8324) | `bench/bench_quality.py` (thresholds `min_ndcg_margin_over` popularity 0.05, tower 0.01), `tests/integration/test_evaluation.py::test_the_cascade_beats_popularity_on_the_ground_truth_slate_metric` |
| The cascade beats popularity on the full-corpus protocol | `tests/integration/test_evaluation.py::test_the_cascade_beats_popularity_on_true_relevance`, `bench/bench_quality.py` (threshold `min_corpus_ndcg_margin_over` popularity 0.15) |
| The cascade retains 1.0007 of the exact full-corpus ranking (gap -0.0004), so the candidate set costs no measurable quality | `bench/bench_quality.py` (threshold `min_upper_bound_retained` 0.8), `tests/integration/test_evaluation.py::test_the_gap_to_the_unservable_upper_bound_is_reported` |
| Every corpus row, `exact_rank` included, is averaged over the same users, and the retention is paired per user | `tests/integration/test_evaluation.py::test_the_upper_bound_is_paired_by_user_and_not_by_position`, `tests/integration/test_evaluation.py::test_no_retention_is_reported_when_no_user_has_both_numbers`, `bench/bench_quality.py` (the corpus table prints a `users` column) |
| The reranker does not beat the tower on the corpus protocol, and neither does the same model scoring the whole catalogue, so the limit is the model | `bench/bench_quality.py` (the corpus table prints all five policies, including `exact_rank`) |
| At 100k items the index is 15.5x faster than brute force for 10 neighbours and 0.61x for the 500 this cascade asks for, crossover near 197k items | `bench/bench_quality.py` (the `scaling` block and its candidate sweep) |
| NDCG, MRR, coverage and diversity are computed correctly | `tests/unit/test_eval_metrics.py::test_ndcg_of_a_perfectly_sorted_list_is_one_and_of_a_reversed_one_is_lower`, `tests/unit/test_eval_metrics.py::test_ndcg_is_in_the_unit_interval`, `tests/unit/test_eval_metrics.py::test_swapping_an_adjacent_pair_into_descending_order_cannot_hurt`, `tests/unit/test_eval_metrics.py::test_catalog_coverage_is_a_share_of_the_catalog` |
| Cold-user and cold-item slices are reported separately | `tests/integration/test_evaluation.py::test_cold_user_slices_are_reported_separately` |
| The feedback simulator is a real click model, not a replay: a better list earns more clicks | `tests/unit/test_simulator.py::test_a_more_relevant_list_earns_more_clicks`, `tests/unit/test_simulator.py::test_position_ctr_decays_down_the_page` |

## Security controls

| Claim | Evidence |
|---|---|
| API keys are SHA-256 digests compared in constant time; protected routes fail closed | `tests/unit/test_security.py::test_no_configured_keys_fails_closed`, `tests/unit/test_security.py::test_every_configured_digest_is_compared_without_early_exit` |
| Every route except `/healthz` and `/readyz` requires a key | `tests/integration/test_cascade_api.py::test_every_post_route_requires_a_key`, `tests/integration/test_cascade_api.py::test_every_get_route_requires_a_key`, `tests/integration/test_cascade_api.py::test_healthz_and_readyz_need_no_key` |
| Admin routes need the admin scope as well | `tests/integration/test_cascade_api.py::test_a_serving_key_cannot_reach_an_admin_route`, `tests/integration/test_demos_cli.py::test_the_demo_shows_every_access_control_refusing_a_request` |
| `user_id`, `k`, `candidate_k`, `ef_search` and lambda are bounded, each with its own error code | `tests/integration/test_cascade_api.py::test_out_of_bounds_parameters_are_refused_with_a_specific_code`, `tests/integration/test_cascade_api.py::test_the_request_schema_rejects_malformed_bodies` |
| The query cost cap refuses combinations that are individually legal but expensive together | `tests/integration/test_cascade_api.py::test_the_cost_cap_refuses_a_pair_that_is_individually_legal`, `tests/integration/test_cascade_api.py::test_the_query_cost_grows_with_both_knobs_and_validates_them` |
| Reindex parameters are bounded twice: by the request schema and by `HnswParams` | `tests/integration/test_cascade_api.py::test_reindex_parameters_are_bounded`, `tests/unit/test_ann.py::test_hnsw_params_are_bounded` |
| Every bundle artifact's digest is verified before a parser sees the bytes | `tests/unit/test_bundle.py::test_flipping_one_byte_in_any_artifact_is_caught_before_it_is_parsed`, `tests/unit/test_ann.py::test_a_tampered_index_never_reaches_the_faiss_parser`, `tests/unit/test_ranker.py::test_a_tampered_model_text_never_reaches_the_lightgbm_parser` |
| `bundle_id` is the hash of the manifest, so a tampered manifest fails its own id check | `tests/unit/test_bundle.py::test_the_bundle_id_is_the_hash_of_the_rest_of_the_manifest`, `tests/unit/test_bundle.py::test_editing_the_manifest_breaks_its_own_id` |
| Nothing in a bundle is pickled | `tests/unit/test_no_pickle.py::test_source_tree_has_no_pickle_family_serialization`, `tests/unit/test_bundle.py::test_nothing_in_a_bundle_is_pickled`, `tests/unit/test_ml_serialization.py::test_pickled_npz_files_are_refused_on_load` |
| The impression log, the catalog and ingested feedback are validated against contracts | `tests/unit/test_contracts.py::test_a_well_formed_log_and_catalog_pass`, `tests/unit/test_contracts.py::test_out_of_range_columns_are_rejected`, `tests/unit/test_generator.py::test_log_and_catalog_satisfy_their_contracts`, `tests/integration/test_cascade_api.py::test_feedback_that_violates_the_contract_is_refused` |
| A contract error never copies row values into the exception or the logs | `tests/unit/test_contracts.py::test_the_error_summarizes_failures_without_copying_any_row_values` |
| The feature snapshot re-validates its own shapes and ranges on every load | `tests/unit/test_features.py::test_the_snapshot_contract_rejects_broken_arrays`, `tests/unit/test_features.py::test_a_non_monotone_seen_index_is_rejected` |
| Robust item statistics damp a shilling attack the naive aggregation falls for (25-27x for one-session accounts) | `tests/integration/test_evaluation.py::test_robust_aggregation_damps_a_shilling_attack_the_naive_one_falls_for`, `bench/bench_quality.py` (thresholds `shilling.fresh_50` and `shilling.fresh_200`) |
| Each user counts once per item and tenure weighting is the only difference between two logs | `tests/unit/test_features.py::test_a_repeated_click_on_the_same_item_counts_once_under_robust_aggregation`, `tests/unit/test_features.py::test_robust_item_counts_weight_a_fresh_account_below_an_established_one` |
| Item CTR is shrunk towards the global rate, so a 1-of-1 click rate cannot stand | `tests/unit/test_features.py::test_the_robust_build_shrinks_item_ctr_towards_the_global_rate` |
| Feedback is metered per source *and* per authenticated key, so rotating the caller-chosen `source` buys no extra budget | `tests/integration/test_cascade_api.py::test_one_noisy_source_cannot_flood_the_statistics`, `tests/integration/test_cascade_api.py::test_rotating_the_source_does_not_buy_a_client_more_feedback_budget`, `tests/integration/test_cascade_api.py::test_two_api_keys_do_not_share_one_feedback_budget`, `tests/integration/test_cascade_api.py::test_a_rate_limited_source_is_told_when_to_retry` |
| An LRU eviction readmits a forgotten key with a full bucket, which is why the bucket that bounds a client lives in its own limiter | `tests/unit/test_ratelimit.py::test_an_evicted_key_is_readmitted_with_a_full_bucket`, `tests/unit/test_ratelimit.py::test_a_throttled_key_that_stays_in_the_map_is_not_reset_by_other_keys` |
| Accepted feedback never feeds training | `tests/unit/test_monitoring.py::test_feedback_updates_the_simulated_ctr_gauge_but_never_training` |
| Privileged actions are appended to a hash-chained audit log before the response returns | `tests/integration/test_cascade_api.py::test_reindex_swaps_the_index_and_records_the_new_digest_in_the_chain`, `tests/integration/test_cascade_api.py::test_the_admin_audit_chain_verifies_after_a_run_of_actions` |
| A refused bundle swap is audited too | `tests/integration/test_cascade_api.py::test_swapping_in_a_tampered_bundle_is_refused_and_the_refusal_is_audited`, `tests/integration/test_cascade_api.py::test_swapping_from_a_directory_that_does_not_exist_is_refused_and_audited` |
| A bundle swap is confined to `bundle_root`: a path outside it is refused before the loader reads anything, so a leaked admin key is not an arbitrary-path oracle | `tests/integration/test_cascade_api.py::test_a_bundle_directory_outside_the_root_is_refused_before_anything_is_loaded`, `tests/integration/test_cascade_api.py::test_the_live_bundle_directory_stays_loadable_from_outside_the_root`, `tests/integration/test_demos_cli.py::test_the_demo_shows_a_bundle_path_outside_the_root_being_refused` |
| Any edit, deletion or reordering of an audit entry breaks the chain, and the break is localized | `tests/unit/test_audit.py::test_editing_an_entry_without_rehashing_breaks_its_own_hash`, `tests/unit/test_audit.py::test_rehashing_an_edited_entry_still_breaks_the_link_to_the_next_one`, `tests/unit/test_audit.py::test_flipping_any_single_entry_is_always_detected`, `tests/integration/test_demos_cli.py::test_the_demo_verifies_the_audit_chain_and_then_breaks_it` |
| User ids are keyed, truncated digests in logs and audit records | `tests/unit/test_audit.py::test_the_digest_is_keyed_so_a_small_id_space_cannot_be_enumerated`, `tests/integration/test_cascade_api.py::test_user_ids_never_reach_the_logs_in_the_clear` |
| Secrets handed to the audit log are redacted before they are written | `tests/unit/test_audit.py::test_secrets_in_details_are_redacted_before_they_are_written` |
| Errors return fixed codes with no exception text, traceback or submitted value | `tests/integration/test_app.py::test_unhandled_exceptions_become_generic_500s`, `tests/integration/test_app.py::test_validation_errors_never_echo_submitted_values` |
| Rate limiting, body-size limits, security headers and explicit-origin CORS | `tests/integration/test_app.py::test_rate_limit_rejects_bursts_and_exempts_probes`, `tests/integration/test_app.py::test_body_size_limit_applies_to_declared_and_streamed_bodies`, `tests/integration/test_app.py::test_healthz_is_public_and_carries_security_headers`, `tests/integration/test_app.py::test_cors_allows_only_configured_origins` |
| Only `.env.example` is committed and it holds no secret values | `tests/unit/test_config.py::test_env_example_documents_every_setting`, `.github/workflows/ci.yml#baseline`, `.github/workflows/ci.yml#gitleaks` |

## Reproducibility and monitoring

| Claim | Evidence |
|---|---|
| The same config produces byte-identical artifacts and an identical metric hash | `tests/integration/test_pipeline.py::test_the_same_config_reproduces_the_data_hash_and_the_metric_hash` |
| The same impression log always produces the same snapshot bytes (the DuckDB non-determinism that was found and fixed) | `tests/unit/test_features.py::test_the_same_log_always_produces_the_same_snapshot_bytes`, `tests/unit/test_features.py::test_the_popularity_order_ignores_differences_below_the_rounding_floor` |
| A different seed changes the data and the metrics | `tests/integration/test_pipeline.py::test_a_different_seed_changes_the_data_and_the_metrics`, `tests/unit/test_generator.py::test_generation_is_a_pure_function_of_the_seed` |
| The manifest records seeds, the log hash, every hyper-parameter, the lockfile hash and package versions | `tests/unit/test_bundle.py::test_the_manifest_records_what_produced_the_bundle`, `tests/integration/test_pipeline.py::test_the_run_manifest_records_the_seed_params_and_package_versions`, `tests/integration/test_pipeline.py::test_the_lockfile_hash_is_recorded_when_a_repo_root_is_given` |
| Item vectors are recomputed from the tower rather than stored, so the canary is end to end | `tests/unit/test_bundle.py::test_item_vectors_are_recomputed_from_the_tower_not_stored`, `tests/integration/test_pipeline.py::test_the_serving_index_matches_the_serving_snapshot_s_vectors` |
| A deliberately corrupted index makes the canary alert within one cycle | `tests/unit/test_canary.py::test_a_corrupted_index_alerts_within_one_cycle`, `tests/integration/test_cascade_api.py::test_a_corrupted_live_index_makes_the_canary_alert_on_the_next_cycle`, `bench/bench_quality.py` (threshold `max_canary_cycles_to_detect` 1) |
| Recall falls monotonically with the corrupted fraction | `tests/unit/test_canary.py::test_recall_falls_monotonically_with_the_corrupted_fraction` |
| A reindex restores recall | `tests/integration/test_cascade_api.py::test_a_reindexed_index_still_passes_the_recall_canary`, `tests/integration/test_demos_cli.py::test_the_demo_shows_the_canary_catching_a_corrupted_index_and_a_reindex_fixing_it` |
| Feature PSI is reported per feature against the bundle's training reference | `tests/unit/test_monitoring.py::test_psi_is_reported_per_feature_once_the_buffer_is_full_enough`, `tests/unit/test_monitoring.py::test_served_traffic_drifts_away_from_the_training_reference` |
| Embedding-norm drift catches a changed item tower | `tests/unit/test_monitoring.py::test_a_changed_item_tower_shows_up_as_embedding_norm_drift` |
| Monitoring buffers are bounded and features are only recomputed for a sampled subset | `tests/unit/test_monitoring.py::test_the_feature_buffer_is_bounded`, `tests/unit/test_monitoring.py::test_feature_rows_are_only_recomputed_for_a_sampled_subset` |
| `stage_latency_seconds`, `degraded_responses_total`, `ann_recall_canary`, `feature_psi`, `embedding_norm_drift` and `catalog_coverage` are exported | `tests/integration/test_cascade_api.py::test_the_metrics_endpoint_carries_both_http_and_cascade_families`, `tests/integration/test_demos_cli.py::test_the_demo_exposes_the_metric_families_a_dashboard_needs` |
| A bundle swap rebinds the monitors and clears the traffic buffers | `tests/unit/test_monitoring.py::test_rebinding_clears_the_buffers_and_re_reads_the_reference` |
| `/readyz` fails while the index does not match the bundle | `tests/integration/test_cascade_api.py::test_readyz_fails_while_the_index_does_not_match_the_bundle` |
| A 1% training slice meets metric thresholds in CI | `tests/integration/test_pipeline.py::test_a_smoke_scale_run_finishes_and_meets_a_recall_threshold`, `.github/workflows/ci.yml#train-smoke` |

## Offline runnability and tooling

| Claim | Evidence |
|---|---|
| The demo runs end to end with no network and no API key, and shows the failure paths | `tests/integration/test_demos_cli.py::test_the_demo_trains_a_bundle_and_serves_it`, `tests/integration/test_demos_cli.py::test_the_demo_shows_a_tight_budget_degrading_rather_than_failing`, `tests/integration/test_demos_cli.py::test_the_demo_shows_a_tampered_bundle_being_refused` |
| No test touches the network | `tests/conftest.py` (the autouse fixture that blocks every non-loopback socket) |
| The CLI trains, verifies a bundle, verifies an audit chain and runs the demo | `tests/integration/test_demos_cli.py::test_train_writes_a_bundle_and_reports_its_metric_hash`, `tests/integration/test_demos_cli.py::test_bundle_verifies_a_bundle_on_disk`, `tests/integration/test_demos_cli.py::test_audit_verify_accepts_a_good_chain_and_can_pin_the_head`, `tests/integration/test_demos_cli.py::test_the_demo_subcommand_prints_the_summary` |
| A tampered bundle or a broken chain is reported as a failure by the CLI, not a crash | `tests/integration/test_demos_cli.py::test_bundle_reports_a_tampered_bundle_as_a_failure`, `tests/integration/test_demos_cli.py::test_audit_verify_reports_a_broken_chain` |
| 750 tests, 98.07% branch coverage against an 85% floor | `.github/workflows/ci.yml#test` |
| Micro-benchmarks run against regression thresholds | `bench/bench_baseline.py`, `.github/workflows/ci.yml#bench` |
| The repo satisfies the shared portfolio baseline, including that every reference in this file resolves | `tests/integration/test_check_baseline.py::test_template_passes_its_own_baseline`, `.github/workflows/ci.yml#baseline` |
| Every `path.py::symbol` citation in the package, the benches and the prose points at something that exists | `tests/unit/test_docs_citations.py::test_every_cited_file_and_symbol_exists`, `tests/unit/test_docs_citations.py::test_the_citation_scan_covers_the_repository` |
| Containers build, run as a non-root uid and pass a health check | `.github/workflows/ci.yml#docker` (not run locally) |
