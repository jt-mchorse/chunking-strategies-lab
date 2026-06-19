# Session History (AI-readable, append-only)

Schema: see .skills/portfolio-memory/SKILL.md

---
session: 2026-05-14T15:35:00Z
duration_min: 50
issue: 1
focus: pin_shared_substrate_corpus_queries_embedding_model
delta:
  files_added: 13
  files_changed: 4
  tests_added: 18
  coverage_pct: 92
  corpus_docs: 5
  queries: 12
context_for_next_session:
  - corpus_is_5_md_files_in_data_corpus_authored_for_this_repo_mit_licensed
  - queries_is_12_verbatim_snippet_queries_in_data_queries_jsonl_each_with_expected_doc_and_expected_snippet
  - canonical_embedding_model_pinned_sentence_transformers_all_minilm_l6_v2_384d
  - hashembedder_default_dim_is_384_so_strategies_tested_under_it_transfer_to_real_embedder_for_3
  - minilmembedder_behind_sbert_extra_keeps_ci_fast_no_model_weights_downloaded_per_run
  - test_each_expected_snippet_is_verbatim_in_its_document_is_the_gate_that_prevents_drift_between_queries_and_corpus
decisions_made: [D-002, D-003]
followups: []
---

---
session: 2026-05-15T18:10Z
duration_min: 75
issue: 2
focus: implement_5_chunking_strategies
delta:
  files_added: 7
  files_changed: 2
  tests_added: 29
  test_pass_rate: "47/47"
context_for_next_session:
  - five_strategies_shipped_fixed_recursive_semantic_late_structure
  - chunk_carries_start_end_offsets_d005_metrics_matrix_3_uses_them_as_join_key
  - late_chunking_special_returns_chunk_plus_vector_pairs_d006
  - per_strategy_modules_d004_strategy_protocol_single_method_seam
  - acceptance_criterion_3_runtime_test_runs_under_5s_per_strategy_real_numbers_pending_operator
  - issue_3_metrics_matrix_unblocked_iterates_strategies_via_protocol
decisions_made: [D-004, D-005, D-006]
followups: []
---

---
session: 2026-05-16T04:29Z
duration_min: 40
issue: 3
focus: retrieval_metrics_matrix_recall_and_snippet_hit_across_5_strategies
delta:
  files_added: 3
  files_changed: 1
  tests_added: 9
  test_pass_rate: "56/56"
context_for_next_session:
  - metrics_module_in_chunking_lab_metrics_py_pure_functions_no_sqlite_d_007
  - evaluate_strategy_takes_strategy_corpus_queries_embedder_ks_returns_retrievalrun
  - recall_at_k_is_doc_level_was_expected_doc_in_top_k_retrieved_chunks_source_doc_ids
  - snippet_hit_at_k_is_answer_faithfulness_proxy_d_008_substring_match_in_top_k_chunks_text
  - late_chunking_routed_through_chunk_with_vectors_not_through_embedder_chunk_text_embedder_must_match
  - scripts_run_matrix_py_single_command_runs_all_5_strategies_writes_one_json_per_strategy_plus_summary_md
  - default_embedder_hash_dep_free_ci_safe_minilm_via_sbert_extra_for_real_numbers
  - summary_md_carries_explicit_disclosure_when_using_hash_embedder_no_fabricated_benchmarks
  - 9_new_tests_56_total_lint_format_clean
  - issue_3_acceptance_single_command_runs_matrix_done_numbers_persisted_to_results_done_test_fixtures_lock_down_tiny_sub_run_done
decisions_made: [D-007, D-008]
followups: []
---

---
session: 2026-05-17T00:30Z
duration_min: 70
issue: 4
focus: comparison_notebook_with_three_charts_plus_honest_takeaways_plus_wall_clock_metric
delta:
  files_added: 4
  files_changed: 5
  tests_added: 7
  test_pass_rate: "63/63"
  benchmarks:
    fixed_size_wall_clock_ms: 20
    recursive_wall_clock_ms: 19
    semantic_wall_clock_ms: 75
    late_chunking_wall_clock_ms: 22
    structure_aware_wall_clock_ms: 19
    embedder: HashEmbedder
    n_queries: 12
context_for_next_session:
  - retrievalrun_now_carries_wall_clock_ms_field_d_009_default_zero_for_backward_compat_with_pre_d009_jsons
  - evaluate_strategy_times_full_chunk_embed_retrieve_pipeline_via_time_perf_counter
  - run_matrix_summary_md_now_includes_wall_clock_column
  - old_20260515_results_jsons_deleted_replaced_with_20260516_runs_carrying_wall_clock_field
  - notebook_extra_named_notebook_matplotlib_jupyter_nbformat_d_010_parallels_d_003
  - comparison_notebook_three_charts_recall_snippet_hit_latency_plus_takeaways_section
  - notebook_built_programmatically_via_notebooks_build_notebook_py_using_nbformat_textwrap_dedent
  - notebook_executed_via_jupyter_nbconvert_inplace_execute_so_committed_file_carries_chart_outputs
  - test_notebook_py_uses_pytest_importorskip_nbformat_so_base_ci_skips_when_extras_not_installed
  - results_loader_picks_latest_run_per_strategy_by_filename_timestamp_canonical_strategy_order_preserved
  - readme_comparison_notebook_subsection_under_benchmarks_results_with_three_step_regenerate_command
  - latency_dominated_by_chunk_count_takeaway_committed_in_notebook_semantic_3x_more_chunks_3x_more_latency
decisions_made: [D-009, D-010]
followups: []
---

---
session: 2026-05-18T15:40Z
duration_min: 20
issue: 9
focus: architecture_doc_recolor_and_layer_sections
delta:
  files_changed: 2  # README.md, docs/architecture.md
  files_added: 0
  tests_added: 0
  test_pass_rate: "63/63"
context_for_next_session:
  - docs_architecture_md_rewritten_integrated_diagram_all_green_includes_notebook_and_wall_clock_axis_plus_four_layer_sections
  - readme_architecture_stub_no_longer_says_shipped_vs_pending_one_line_pointer
  - pending_section_removed_every_layer_in_section_2_has_shipped
  - mermaid_labels_with_parens_fully_double_quoted_same_lint_as_other_repos_this_session
  - no_new_d_entry_references_d_002_through_d_010
decisions_made: []
followups: []
---

---
session: 2026-05-18T19:39Z
duration_min: 15
issue: 11
focus: snapshot_test_locks_results_summary_md_to_render_summary_over_committed_jsons
delta:
  files_added: 1   # tests/test_summary_snapshot.py
  files_changed: 0
  tests_added: 7   # 1 byte-eq snapshot + 1 strategy-set guard + 5 parametrized loader checks
  test_pass_rate: "70/70"
context_for_next_session:
  - snapshot_loads_committed_results_jsons_reconstructs_retrieval_run_with_empty_per_query_tuple_renderer_does_not_use_per_query
  - strategy_order_locked_fixed_size_recursive_semantic_late_chunking_structure_aware_matches_run_matrix_build_strategies
  - parametrized_loader_check_per_strategy_guards_against_new_required_retrieval_run_field_added_without_migration
  - tamper_verified_fixed_size_recall_at_1_0_333_to_0_999_test_fired_then_reverted
  - pattern_parallel_to_three_other_snapshot_tests_landed_today_cost_optimizer_prompt_regression_rag_kit
  - no_new_d_entry_enforces_handoff_section_10_no_fabricated_benchmarks
decisions_made: []
followups: []
---

---
session: 2026-05-19T04:00Z
duration_min: 30
issue: 11
focus: unblock_pr_12_un_ignore_canonical_fixtures
delta:
  files_changed: 4   # .gitignore, README.md, scripts/run_matrix.py, tests/test_metrics.py
  files_added: 6    # 5 canonical__*.json + summary.md newly tracked
  files_renamed: 5   # 20260516T162215__*.json → canonical__*.json
  tests_added: 1    # test_run_matrix_default_writes_timestamped_scratch
  test_pass_rate: "71/71"
context_for_next_session:
  - root_cause_gitignore_results_blanket_meant_snapshot_fixtures_never_landed_in_ci_checkout
  - fix_un_ignored_canonical_glob_plus_summary_md_kept_timestamped_pattern_as_gitignored_scratch
  - run_matrix_now_takes_canonical_out_flag_default_writes_timestamped_scratch_pair_summary_too
  - canonical_filenames_use_double_underscore_separator_so_existing_test_split_logic_works_unchanged
  - new_test_locks_default_no_tracked_filenames_written_complements_existing_canonical_path_test
  - tamper_verification_recall_at_5_substitution_in_summary_md_still_fires_snapshot
  - no_new_d_entry_pure_hygiene_fix
decisions_made: []
followups: []
---

---
session: 2026-05-19T15:20Z
duration_min: 30
issue: 13
focus: snapshot_test_locks_comparison_ipynb_cell_sources_to_build_notebook_py
delta:
  files_added: 1   # tests/test_build_notebook_snapshot.py
  files_changed: 1 # notebooks/_build_notebook.py refactor + _LOAD_CELL print() reformat
  tests_added: 11  # 1 cell-count guard + 10 parametrized per-cell (cell_type, source)
  test_pass_rate: "82/82"
context_for_next_session:
  - build_notebook_function_now_pure_returns_notebooknode_main_writes_nbformat
  - snapshot_test_imports_build_notebook_via_sys_path_insert_notebooks_dir
  - cell_signature_is_cell_type_plus_source_rstrip_papers_over_nbformat_round_trip_newline_asymmetry
  - outputs_execution_count_metadata_intentionally_ignored_pixel_content_env_dep_already_covered_by_test_notebook_executed_outputs_exist
  - parametrize_index_range_computed_at_collection_time_via_len_build_notebook_cells_adding_a_cell_grows_test_set_automatically
  - first_run_caught_real_drift_load_cell_print_was_single_line_in_build_script_multi_line_in_committed_ipynb_fixed_build_script_not_notebook
  - tamper_verified_intro_edit_fires_cell_0_then_reverted
  - pattern_parallel_to_summary_snapshot_in_this_repo_and_eval_aggregator_snapshots_in_sister_repos
  - no_new_d_entry_d_009_d_010_still_govern
decisions_made: []
followups: []
---

---
session: 2026-05-20T03:21Z
duration_min: 20
issue: 15
focus: public_surface_snapshot_test_locks_chunking_lab_top_level_init
delta:
  files_added: 1   # tests/test_public_surface.py
  files_changed: 1   # chunking_lab/__init__.py (+__version__)
  tests_added: 8   # 4 standalone + 4 parametrized submodule anchors
  test_pass_rate: "90/90"
context_for_next_session:
  - chunking_lab_now_publishes_dunder_version_str_0_0_1
  - readme_quickstart_test_unions_both_snippets_lines_74_and_94_six_names
  - metrics_intentionally_dotted_path_only_excluded_from_submodule_anchors_no_decision_change
  - tamper_verified_three_axes_bad_version_drop_document_alias_rename_canonical_embedding_model
  - portable_pattern_sixth_strike_remaining_python_async_llm_pipelines_and_mcp_python_example
decisions_made: []
followups: []
---

---
session: 2026-05-21T19:16Z
duration_min: 22
issue: 17
focus: scripts_capture_demo_sh_two_surface_60s_driver_plus_smoke_test
delta:
  files_added: 2   # scripts/capture_demo.sh, tests/test_capture_demo_smoke.py
  files_changed: 1 # README.md (Demo section pending placeholder → real invocation)
  tests_added: 3
  test_pass_rate: "93/93"
context_for_next_session:
  - seventh_repo_to_land_capture_demo_sh_pattern_this_week
  - two_surfaces_run_matrix_results_dir_tmp_then_cat_ts_summary_md_chosen_for_runtime_under_one_second
  - hash_embedder_deliberate_for_tempo_and_hermeticity_quality_claims_stay_in_canonical_results_summary_md_and_comparison_notebook
  - smoke_test_pins_all_five_strategy_names_in_matrix_output_plus_per_line_recall_snippet_hit_wall_clock_keys
  - smoke_test_pins_summary_markdown_header_belt_and_braces_with_test_summary_snapshot
  - per_run_tempdir_via_mktemp_d_trap_exit_int_term
  - no_new_d_entry_d_008_snippet_hit_metric_already_governs_this_is_pure_glue
decisions_made: []
followups: []
---

---
session: 2026-05-22T03:25Z
duration_min: 30
issue: 19
focus: enforce_late_chunking_embedder_consistency_at_runner
delta:
  files_changed: 1   # chunking_lab/metrics.py
  files_added: 0     # tests appended to existing tests/test_metrics.py
  tests_added: 4
  test_pass_rate: "97/97"
decisions_made: [D-011]
context_for_next_session:
  - silent_footgun_in_evaluate_strategy_when_late_chunking_strategy_embedder_diverges_from_runner_embedder_cosine_meaningless_no_exception
  - enforcement_compares_by_underscore_embedder_model_name_helper_not_python_identity_so_two_hashembedder_instances_pass_through_correctly
  - check_lives_in_underscore_check_late_chunking_embedder_consistency_runs_before_materialize_vectors_so_no_partial_work_done_before_failure
  - error_message_names_the_strategy_class_and_d_011_so_caller_can_find_rationale_without_reading_the_source
  - test_doubles_use_named_embedder_class_inner_hashembedder_only_differs_in_model_name_attribute_avoids_pulling_sbert_into_ci
  - non_late_strategy_isolation_test_proves_check_is_a_no_op_for_fixed_recursive_semantic_structure_so_no_regression_for_4_of_5_strategies
  - d_011_added_to_both_core_decisions_files_separate_commit_per_protocol
followups: []
---

---
session: 2026-05-23T04:05Z
duration_min: 25
issue: 23
focus: readme_drift_fix_three_this_pr_sites_d_011_omission_plus_snapshot_lock
decisions_made: []
delta:
  files_changed: 1   # README.md
  files_added: 1     # tests/test_readme_snapshot.py
  tests_added: 6
  test_pass_rate: "103/103"
context_for_next_session:
  - this_was_last_portfolio_repo_without_readme_snapshot_lock_now_at_12_of_12
  - drift_fix_l19_pre_shipping_paragraph_l41_d_011_omission_l59_l109_this_pr_section_headers
  - active_decision_range_upper_bound_test_anchors_to_memory_core_decisions_ai_md_loud_failure_when_a_new_d_lands_without_readme_updating
  - portfolio_pattern_two_locks_now_complete_arch_doc_lock_12_of_12_readme_lock_12_of_12
followups: []
---

---
session: 2026-05-24T03:46Z
duration_min: 20
issue: 25
focus: run_matrix_strategy_filter_for_iterative_single_strategy_dev
delta:
  files_changed: 1   # scripts/run_matrix.py
  files_added: 1     # tests/test_run_matrix_strategy_filter.py
  tests_added: 6
  test_pass_rate: "119/119"
decisions_made: []
context_for_next_session:
  - run_matrix_ran_all_five_strategies_every_invocation_no_way_to_iterate_one_strategy_without_clobbering_other_four_under_canonical_out
  - strategy_filter_argparse_choices_matches_each_strategy_name_attr_fixed_size_recursive_semantic_late_chunking_structure_aware_same_naming_as_canonical_files
  - no_summary_when_filtered_rule_keeps_partial_row_failure_from_poisoning_the_snapshot_lock_and_misleading_the_canonical_aggregate
  - tests_pin_does_not_clobber_other_canonical_files_via_pre_seeded_sentinel_bytes
  - parallel_pattern_to_rag_production_kit_issue_32_suite_filter_landed_earlier_this_session
  - sixth_in_night_session_multi_issue_loop
followups: []
---

---
session: 2026-05-25T00:55Z
duration_min: 15
issue: 27
focus: evaluate_strategy_validates_ks_per_element_at_function_entry
delta:
  files_changed: 1   # chunking_lab/metrics.py
  files_added: 0
  tests_added: 7   # empty + zero + negative + mixed + 3 parametrized positive acceptance
  test_pass_rate: "126_passed"
decisions_made: []
context_for_next_session:
  - evaluate_strategy_validated_no_ks_at_entry_max_k_max_ks_if_ks_else_5_silently_treated_empty_as_max_5_with_empty_recall_dict_compute
  - non_positive_k_flowed_through_retrieved_docs_k_slicing_recall_at_zero_always_zero_recall_at_minus_one_all_but_last_silent_miscount_wrong_number_not_absent_number
  - guard_at_function_entry_collects_all_offenders_one_pass_sorted_form_message_shape_matches_emb_shootout_pr_28_sister_fix
  - empty_ks_raises_separately_before_per_element_check_so_messages_stay_specific_to_each_bug_class
  - mirrors_embedding_model_shootout_pr_28_run_sweep_k_values_guard_two_retrieval_comparator_repos_now_defend_result_jsons_consistently
  - tests_pin_empty_zero_negative_mixed_with_full_set_in_message_plus_parametrize_positive_acceptance_over_one_two_four_element_shapes
  - test_helper_eval_with_ks_centralizes_evaluate_strategy_construction_each_test_only_mutates_ks_under_test
  - test_count_chunking_lab_now_126_was_119_after_25_added_7_new_collected_cases
  - fifth_phase_bc_target_in_180_min_day_session_after_phase_a_5_pr_merge_plus_four_prior_phase_bc_targets
followups: []
---

---
session: 2026-05-25T06:55Z
duration_min: 25
issue: 29
focus: strategy_dataclasses_and_hashembedder_isinstance_int_guards_extend_sign_only
delta:
  files_changed: 6   # fixed.py late.py recursive.py semantic.py embedder.py test_strategies.py
  files_added: 0
  tests_added: 35   # 7 parametrize tables x 5 bad values = 35 new collected cases
  test_pass_rate: "162_passed"
decisions_made: []
context_for_next_session:
  - second_pr_in_chunking_lab_tonight_first_via_phase_a_fixup_merge_of_28_sign_only_ks_per_element_in_evaluate_strategy
  - five_constructors_sign_only_on_chunk_chars_overlap_chars_min_max_chunk_chars_dim_non_int_silently_propagates
  - chunk_chars_nan_text_start_nan_typeerror_cryptic_deep_in_chunking_loop
  - overlap_chars_nan_passes_both_sign_only_and_overlap_geq_chunk_check_then_stride_nan_then_start_plus_nan_while_start_lt_len_undefined_infinite_loop_in_worst_case
  - fractional_chunk_chars_silently_truncates_via_slicing_int_coercion_producing_wrong_sized_chunks
  - hashembedder_dim_non_int_reached_mod_8_check_misleading_must_be_multiple_of_8_message_rather_than_must_be_an_int
  - tightened_five_constructors_to_isinstance_x_int_with_explicit_bool_exclusion_python_bool_subclasses_int
  - existing_message_matchers_chunk_chars_overlap_chars_etc_survive_unchanged_no_pre_existing_test_updates_needed
  - tenth_phase_bc_target_in_360_min_night_session_after_prompt_regression_35_agent_orchestration_29_mcp_cookbook_32_nextjs_streaming_24_ai_app_integration_tests_24_llm_eval_harness_42_llm_cost_optimizer_36_rag_production_kit_38_embedding_model_shootout_31
  - portfolio_contract_tightening_sweep_now_at_ten_phase_bc_prs_plus_seven_phase_a_fixup_merges_session_tonight
followups: []
---

---
session: 2026-05-26T03:35Z
duration_min: 25
issue: 31
focus: structure_aware_strategy_isinstance_int_bool_reject_completes_29_sweep
delta:
  files_changed: 2   # chunking_lab/strategies/structure.py, tests/test_strategies.py
  files_added: 0
  tests_added: 28   # 6 parametrize blocks: 5+5+6+5+4+3 collected cases
  test_pass_rate: "190_passed"
decisions_made: []
context_for_next_session:
  - 31_completes_the_29_sweep_structure_aware_was_the_only_strategy_constructor_29_missed
  - max_heading_level_added_isinstance_int_plus_bool_reject_above_existing_range_check_preserved_verbatim
  - max_chunk_chars_added_isinstance_int_plus_bool_reject_above_existing_sign_only_le_zero_check_preserved_verbatim
  - error_messages_match_29_shape_exactly_max_field_must_be_an_int_got_repr_so_existing_matchers_carry_over
  - silent_failure_mode_one_max_heading_level_true_silently_bound_self_value_true_downstream_len_m_group_one_le_true_behaves_as_le_one_strategy_degraded_to_splitting_only_on_top_level_hash_silently_semantic_bug_with_no_error
  - silent_failure_mode_two_max_heading_level_2_0_silently_bound_as_float_downstream_comparisons_still_work_strategy_proceeds_with_wrong_typed_field
  - silent_failure_mode_three_max_chunk_chars_true_silently_bound_then_fixedsize_fallback_caught_29_tightened_check_but_misleading_internal_site_error
  - silent_failure_mode_four_max_chunk_chars_nan_or_inf_or_4000_5_silently_bound_surfaces_as_misleading_internal_site_error_from_fixedsize_fallback
  - test_strategy_six_parametrize_blocks_following_existing_bad_int_pattern_two_type_matrices_two_acceptance_matrices_two_preservation_pins_for_existing_range_and_positive_errors
  - test_count_chunking_lab_now_190_was_162_after_29_added_28_new_collected_cases
  - third_phase_bc_target_in_360_min_night_session_after_prompt_regression_37_extend_to_hash_embedder_ngram_plus_canonical_embedding_finiteness
followups: []
---

---
session: 2026-05-26T19:50Z
duration_min: 15
issue: 33
focus: add_chunking_lab_io_utils_atomic_write_text_route_run_matrix_through_it
delta:
  files_added: 2
  files_changed: 1
  tests_added: 7  # 6 unit + 1 integration
  test_pass_rate: "197_passed"
decisions_made: [D-012]
context_for_next_session:
  - fifth_phase_bc_issue_of_today_day_session_canonical_fixture_write_atomicity
  - two_production_sites_closed_run_matrix_per_strategy_json_and_summary_md
  - decision_d_012_atomic_write_helper_lives_in_chunking_lab_io_utils_module_matches_portfolio_standard
  - portfolio_atomic_write_coverage_now_at_ten_of_twelve_repos_remaining_vector_search_at_scale_five_sites_plus_nextjs_streaming_no_write_paths
  - load_bearing_invariant_canonical_fixture_overwrite_failure_preserves_committed_published_numbers_property_path_write_text_could_never_offer_so_published_numbers_no_longer_corruptible_via_a_killed_regen
  - elapsed_approx_55_min_of_180_min_budget_room_for_one_more_issue_to_reach_12_of_12_saturation
followups: []
---

---
session: 2026-05-27T04:00Z
duration_min: 3
issue: 35
focus: contributing_md_cap_wording_propagation_d_008_d_004
delta:
  files_changed: 1   # CONTRIBUTING.md
  tests_added: 0
context_for_next_session:
  - portfolio_wide_propagation_of_portfolio_ops_3_contributing_seed_update_to_d_008_caps_180_360_min_plus_d_004_phase_a_auto_merge_wording_one_pr_per_repo
  - pre_d_008_stale_60_min_cap_wording_originated_in_bootstrap_template_init_portfolio_repo_sh_line_102
decisions_made: []
followups: []
---

---
session: 2026-06-01T23:20Z
duration_min: 17
issue: 37
focus: queries_validate_collecting_mode_lint_jsonl_pre_flight_pattern_propagation_4_of_n_first_repo_with_cross_file_corpus_dir_invariant
phase: day_session_phase_b_iteration_2
delta:
  files_changed: 2   # README.md, docs/architecture.md
  files_added: 2     # chunking_lab/validate.py, tests/test_validate.py
  tests_added: 33    # 1 shipped happy + 1 with corpus_dir + 1 multi-finding + 14 parametrized codes + 1 duplicate + 1 empty + 1 blank-line + 2 corpus-dir variants + 2 missing-file + 1 to_dict + 1 frozen + 1 required-fields + 6 cli end-to-end
context_for_next_session:
  - validate_queries_path_corpus_dir_none_walks_jsonl_in_collecting_mode_returns_validationreport_with_ok_property_true_iff_zero_findings_and_n_valid_gt_zero_same_shape_as_eval_harness_emb_shootout_prompt_regression_pattern_now_in_four_repos
  - sixteen_finding_codes_malformed_json_not_an_object_four_missing_underscore_field_four_non_string_underscore_field_four_empty_underscore_field_duplicate_id_expected_doc_not_found_empty_cross_file_corpus_dir_check_only_when_corpus_dir_passed_first_repo_in_validate_propagation_arc_with_cross_file_invariant
  - python_dash_m_chunking_lab_dot_validate_path_dash_dash_corpus_dir_dash_dash_json_no_console_script_entry_point_per_d_004_minimalism_argparse_only_exit_codes_zero_clean_one_findings_two_io_uniform_with_sister_validators
  - cross_file_invariant_expected_doc_not_found_catches_typo_d_doc_references_that_silently_invalidate_recall_run_completes_number_becomes_meaningless_highest_leverage_check_in_this_validator_shipped_queries_jsonl_validates_clean_against_data_corpus_dir
  - architecture_doc_lock_passes_structurally_no_known_shipped_issues_tuple_in_this_repo_unlike_emb_shootout_one_pre_flight_paragraph_added_under_section_1_pinned_substrate_plus_substrate_bullet_in_where_to_look_next_extended_with_validate_py_pound_37
  - readme_architecture_tree_gained_validate_py_line_under_chunking_lab_with_pound_37_annotation_no_prose_numerics_change_d_002_d_012_range_lock_unchanged_since_no_new_d_nnn_added
  - live_run_against_data_queries_jsonl_real_12_row_shipped_substrate_exits_zero_one_pass_validates_no_false_positives_on_healthy_queries_dash_dash_corpus_dir_data_corpus_also_exits_zero_confirms_cross_file_invariant_holds_on_shipped_substrate
  - 230_of_230_pytest_pass_ruff_check_plus_format_check_clean_pattern_propagation_now_four_repos_validate_eval_harness_prompt_regression_emb_shootout_chunking_lab
  - one_test_bug_caught_during_development_first_cli_malformed_assertion_expected_valid_2_but_actual_was_valid_1_because_row_3_with_empty_question_is_well_formed_json_but_not_counted_as_valid_per_n_valid_semantics_corrected_test_not_code
  - deferred_per_plan_top_level_chunking_lab_console_script_d_004_minimalism_kept_snippet_verification_against_expected_doc_contents_out_of_scope_auto_fix_rewrite_mode_out_of_scope_corpus_md_validator_out_of_scope_markdown_is_freeform
  - next_iteration_candidate_repos_vector_search_at_scale_python_async_llm_pipelines_agent_orchestration_platform_mcp_server_cookbook_nextjs_streaming_ai_app_integration_tests_in_build_sequence_order_check_each_for_fail_fast_jsonl_or_data_loader_to_continue_validate_pattern_propagation
decisions_made: []
followups: []
---

---
session: 2026-06-17T19:35Z
duration_min: 10
issue: 39
focus: workflow_yaml_parseability_lock_propagation_from_portfolio_ops_30
phase: day_session_phase_b_iteration_3
delta:
  files_added: 1   # tests/test_workflows_yaml_parseable.py
  files_changed: 1 # pyproject.toml ([dev] adds pyyaml>=6.0)
  tests_added: 3   # 1 smoke + 1 parse + 1 jobs for ci.yml
context_for_next_session:
  - third_hop_in_propagation_arc_after_llm_eval_harness_60_61_and_rag_production_kit_52_53_for_portfolio_ops_30_31_yaml_parseability_lock
  - only_one_workflow_file_ci_yml_so_total_parametrize_count_lower_than_llm_eval_harness_or_rag_production_kit_three_versus_five
  - pyyaml_added_to_dev_extras_short_inline_comment_pointing_back_to_39_and_portfolio_ops_27
  - test_count_pre_branch_233_post_branch_236_no_regressions
  - nine_remaining_portfolio_repos_still_need_lock_separate_issues_per_repo
decisions_made: []
followups: []
---

---
session: 2026-06-18T03:18Z
duration_min: 20
issue: 41
focus: workflow_timeout_minutes_lock_propagation_third_hop
phase: night_session_phase_b_iteration_2
delta:
  files_added: 1   # tests/test_workflows_timeout_minutes.py
  files_changed: 1 # .github/workflows/ci.yml
  tests_added: 10  # 1 smoke + 3 jobs * 3 invariants
context_for_next_session:
  - third_propagation_hop_of_timeout_minutes_lock_after_llm_eval_harness_63_and_rag_production_kit_55
  - ci_yml_three_jobs_lint_test_memory_check_all_get_15_min_uniform_no_outliers_warranted_at_current_workload
  - test_file_modeled_on_canonical_first_hop_llm_eval_harness_three_parametrized_invariants_split_so_each_failure_mode_surfaces_separately
  - lock_test_failure_messages_explain_silent_failure_mode_6_hour_quota_burn_on_hung_jobs_plus_propagation_arc_back_to_upstream_yaml_parseable_lock_in_this_repo
  - audit_phase_a_py_will_drop_chunking_strategies_lab_from_missing_timeout_finding_set_after_this_pr_merges
  - pre_branch_pytest_count_unchanged_post_branch_plus_10_full_suite_clean_ruff_check_plus_format_check_clean
decisions_made: []
followups: []
---

---
session: 2026-06-18T15:36Z
duration_min: 10
issue: 43
focus: workflow_concurrency_guard_plus_lock_test_propagation_from_llm_eval_harness_64
phase: day_session_phase_b_iteration_4
delta:
  files_added: 1   # tests/test_workflows_concurrency.py
  files_changed: 1 # .github/workflows/ci.yml
  tests_added: 7   # 1 smoke + 3 parametrized x 1 workflow
context_for_next_session:
  - fourth_per_repo_hop_of_concurrency_lock_propagation_after_llm_eval_harness_64_llm_cost_optimizer_60_rag_production_kit_56
  - ci_yml_group_ci_dollar_github_ref_single_workflow_repo
  - lock_test_copy_with_sed_docstring_swap_origin_56_to_43_companion_refs_unparenthesized_to_avoid_stale_issue_link
  - test_count_240_to_247_seven_new_full_pytest_clean_ruff_check_plus_format_check_clean
decisions_made: []
followups: [#43]
---

---
session: 2026-06-19T04:10Z
duration_min: 22
issue: 45
focus: chunking_lab_validate_cli_out_flag_for_sink_parity_propagation_of_llm_eval_harness_66
phase: night_session_phase_b_iteration_3
delta:
  files_changed: 2 # chunking_lab/validate.py + tests/test_validate.py
  tests_added: 6   # 2 modes (human+json) + parent-dir + atomic-overwrite + stderr + exit-2
context_for_next_session:
  - sibling_propagation_of_llm_eval_harness_66_pr_67_validate_cli_out_flag_for_sink_parity_now_lands_in_chunking_strategies_lab_at_the_same_shape_recipe_identical
  - main_routes_through_chunking_lab_io_utils_atomic_write_text_when_out_set_falls_back_to_sys_stdout_write_rendered_when_not_findings_still_print_to_stderr_in_human_mode_regardless_of_out
  - exit_2_file_not_found_path_raises_before_rendering_so_out_leaves_no_zero_byte_sentinel_a_ci_step_could_mistake_for_ran_successfully_locked_by_test_cli_out_not_written_on_file_not_found
  - trailing_newline_parity_both_renderers_add_newline_to_rendered_string_out_writes_full_string_atomically_stdout_uses_sys_stdout_write_rendered
  - test_count_247_to_253_six_new_full_pytest_clean_ruff_check_plus_format_check_clean
  - canonical_propagation_chain_now_pkg_eval_harness_run_list_36_diff_diff_json_validate_66_chunking_lab_validate_45_three_repos_one_shape_atomic_write_text_via_io_utils
  - readme_doesnt_currently_document_the_validate_cli_only_a_one_line_architecture_mention_no_doc_update_needed_this_pr
decisions_made: []
followups: []
---
