# Core Decisions (AI-readable, YAML, append-only)
# Schema: see .skills/portfolio-memory/SKILL.md

- id: D-001
  date: 2026-05-10
  decision: scope_per_portfolio_handoff_section_2
  rationale: locked_scope_prevents_drift
  alternatives_rejected: []
  reversibility: expensive
  related_issues: []
  superseded_by: null

- id: D-002
  date: 2026-05-14
  decision: pin_substrate_5_md_corpus_12_verbatim_queries_minilm_l6_v2_embedder
  rationale: chunking_strategy_comparison_only_makes_sense_when_corpus_queries_model_are_fixed_so_strategy_is_the_only_variable
  alternatives_rejected: [use_external_corpus_with_separate_license, leave_substrate_per_strategy_implementor]
  reversibility: expensive
  related_issues: [1, 2, 3]
  superseded_by: null

- id: D-003
  date: 2026-05-14
  decision: minilm_embedder_behind_sbert_optional_extra_hashembedder_dep_free_default
  rationale: ci_should_not_download_model_weights_per_run_but_users_can_install_real_embedder_for_3_metrics
  alternatives_rejected: [bundle_sentence_transformers_as_required_dep, ship_only_hashembedder_no_real_path]
  reversibility: cheap
  related_issues: [1, 3]
  superseded_by: null

- id: D-004
  date: 2026-05-15
  decision: each_strategy_is_standalone_module_under_strategies_subpackage_single_method_protocol_seam
  rationale: cookbook_principle_reader_can_copy_one_strategy_without_dragging_in_siblings_protocol_keeps_metrics_matrix_uniform
  alternatives_rejected: [single_file_with_all_strategies, abstract_base_class_inheritance, sklearn_style_estimators]
  reversibility: cheap
  related_issues: [2, 3]
  superseded_by: null

- id: D-005
  date: 2026-05-15
  decision: chunk_carries_start_offset_end_offset_back_into_source_text
  rationale: metrics_matrix_3_attributes_retrieved_chunks_to_documents_without_re_tokenizing_offsets_are_the_universal_join_key
  alternatives_rejected: [chunk_id_only_no_offsets, separate_offset_index_keyed_by_chunk_id]
  reversibility: cheap
  related_issues: [2, 3]
  superseded_by: null

- id: D-006
  date: 2026-05-15
  decision: late_chunking_returns_chunk_plus_vector_pairs_other_strategies_return_chunks_only
  rationale: late_chunking_vectors_derive_from_document_level_context_so_caller_cant_recompute_them_from_chunk_text_alone
  alternatives_rejected: [late_chunking_returns_chunks_only_lossy, all_strategies_return_chunk_plus_vector_wasted_compute_for_4_of_5]
  reversibility: cheap
  related_issues: [2, 3]
  superseded_by: null

- id: D-007
  date: 2026-05-16
  decision: metrics_are_pure_functions_over_retrievalrun_no_sqlite
  rationale: ci_runners_ephemeral_one_current_vs_one_baseline_pattern_matches_llm_eval_harness_d_010
  alternatives_rejected: [persist_runs_to_sqlite_in_ci, ship_db_as_artifact, accumulate_results_in_module_global]
  reversibility: cheap
  related_issues: [3]
  superseded_by: null

- id: D-008
  date: 2026-05-16
  decision: snippet_hit_at_k_is_the_answer_faithfulness_proxy_structural_not_semantic
  rationale: cheap_hermetic_gates_strategies_that_fragment_passages_llm_judge_faithfulness_lives_in_eval_harness_consumer
  alternatives_rejected: [llm_judge_in_this_layer_adds_eval_harness_dep, no_faithfulness_metric_at_all_misses_the_fragmentation_failure_mode]
  reversibility: cheap
  related_issues: [3]
  superseded_by: null

- id: D-009
  date: 2026-05-17
  decision: retrievalrun_carries_wall_clock_ms_field_measured_by_evaluate_strategy_default_zero_for_backward_compat
  rationale: notebook_chart_3_is_latency_per_strategy_acceptance_criteria_on_4_evaluate_strategy_is_the_only_place_with_visibility_into_the_full_chunk_embed_retrieve_pipeline_default_zero_so_pre_d_009_jsons_still_load
  alternatives_rejected: [time_each_phase_separately_too_much_surface_for_the_chart_we_need, time_only_chunking_misses_embedding_cost_which_dominates, leave_to_caller_couples_chart_consumers_to_strategy_internals]
  reversibility: cheap
  related_issues: [4]
  superseded_by: null

- id: D-010
  date: 2026-05-17
  decision: notebook_extra_named_notebook_matplotlib_jupyter_nbformat_parallel_to_d_003_sbert_pattern
  rationale: comparison_notebook_is_optional_visualization_surface_base_install_stays_dep_free_runner_works_without_the_notebook_test_uses_importorskip_so_base_ci_passes
  alternatives_rejected: [matplotlib_in_base_install_breaks_dep_free_default_and_pulls_numpy_into_every_ci_run, render_charts_as_static_pngs_via_a_script_loses_the_notebook_artifact_that_4_asked_for, separate_extras_for_matplotlib_and_jupyter_unnecessary_split]
  reversibility: cheap
  related_issues: [4]
  superseded_by: null

- id: D-011
  date: 2026-05-22
  decision: evaluate_strategy_enforces_late_chunking_embedder_consistency_by_model_name
  rationale: documented_constraint_in_materialize_vectors_was_silent_at_runtime_so_mismatched_spaces_produced_garbage_recall_curves_without_warning_repos_credibility_depends_on_not_having_silent_numerical_quality_bugs_in_a_strategy
  alternatives_rejected: [keep_documented_only_status_quo_silent_bug_persists, infer_strategy_embedder_from_runner_breaks_construction_surface_contract, identity_check_strategy_embedder_is_runner_embedder_too_strict_rejects_two_hashembedder_instances]
  reversibility: cheap
  related_issues: [19]
  superseded_by: null

- id: D-012
  date: 2026-05-26
  decision: atomic_write_helper_lives_in_package_level_chunking_lab_io_utils_module_following_portfolio_standard
  rationale: matches_rag_kit_eval_harness_emb_shootout_prompt_regression_async_pipelines_centralizes_test_surface_one_os_replace_to_monkey_patch
  alternatives_rejected: [file_private_helper_per_script, inline_pattern_at_each_call_site]
  reversibility: cheap
  related_issues: [#33]
  superseded_by: null
