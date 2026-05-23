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
