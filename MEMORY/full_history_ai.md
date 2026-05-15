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
