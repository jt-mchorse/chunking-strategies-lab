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
