# An LLM Evaluation Harness That Won't Bite You Later

## Why a dedicated harness

LLM behavior drifts. Model upgrades change response format, prompt edits change topic coverage, retrieval changes change which documents the model sees. An evaluation harness is the cheap insurance against silent regressions — the difference between catching a 12% recall drop in CI and catching it in a customer support escalation.

A good harness sits between unit tests (too narrow) and full integration tests (too expensive to run on every PR). It runs a fixed set of representative queries, scores the responses, and surfaces the deltas in a pull request comment.

## Components

The minimum viable harness has four pieces:

1. **A golden dataset format.** Versioned JSONL where each row contains the input, the expected outputs (as typed objects, not just strings — so a future regex matcher can be added without rewriting the corpus), and any metadata.

2. **An LLM-as-judge wrapper** with calibration against a small human-labeled set. Calibration is what keeps the judge honest — without it, the harness will accept the failure modes the judge model itself has.

3. **A regression runner** that diffs current results against a stored baseline. Numbers without diffs are noise; the harness's value is in the diff.

4. **A CI integration** that posts the deltas as a PR comment. Pushing this into Slack or email loses the in-context review experience.

## What to score

The metrics you measure depend on the workflow under test. For retrieval-augmented generation:

- **Faithfulness:** does every claim in the response cite a chunk that supports it?
- **Recall@k:** is the gold passage retrieved in the top-k?
- **Answer correctness:** by judge or by exact-match-on-canonical-answer when one exists.

For agent workflows:

- **Tool-call correctness:** the agent called the right tool with the right arguments.
- **Step-count efficiency:** did the agent converge in a reasonable number of steps?
- **Final-answer correctness:** judge-scored against the expected answer.

## Anti-patterns

Three anti-patterns to avoid:

1. **No baseline.** A score in isolation is meaningless; the diff is the signal.
2. **Hand-edited expected outputs.** If the expected output is wrong, the test scores nothing useful. Lock the corpus and treat changes to it as deliberate.
3. **Fabricated numbers.** If the eval didn't run, the README says "benchmarks pending," not a made-up table. Discovery of fabricated numbers destroys trust faster than any other failure mode.
