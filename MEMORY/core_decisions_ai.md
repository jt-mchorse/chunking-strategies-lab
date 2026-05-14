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
