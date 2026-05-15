# Prompt Caching with Anthropic Models

## What it does

Anthropic's prompt caching feature lets you mark prefix segments of a request as cacheable. When a subsequent request shares that prefix, the cached portion is reused instead of re-tokenized and re-processed end-to-end. The result is a substantial reduction in input-token cost on the cached segment — published rates put the cache-read multiplier at roughly 0.10× the normal input rate, with a one-time write cost on the first request.

The caller marks segments with `cache_control: {"type": "ephemeral"}` on the system prompt, on a tool definition block, or on a leading subset of messages. Anthropic's server matches incoming requests against recently-written cache entries by prefix hash and reuses them when the prefix matches exactly.

## Where the savings come from

Three patterns benefit most:

1. **Large system prompts shared across many calls.** Customer-support assistants, code-review agents, and any workflow that re-injects the same large context per call. The system prompt is identical across calls; only the user message changes. Caching the system prompt makes the marginal cost of a call dominated by the user message.

2. **Long tool definitions.** Agents that expose many tools accumulate long tool-definition blocks. Marking the tool list as cacheable amortizes its cost across every tool-use turn.

3. **Long document prefixes for question-answering.** When a workflow asks several questions about the same long document, prefix-caching the document makes each subsequent question close to free for the document portion.

## Constraints and gotchas

Cache entries have a short TTL — measured in minutes — and silently miss if the prefix doesn't exactly match the cached version. A trailing whitespace difference, a different order of tool definitions, or a model-version change all defeat the cache. The wrapper that integrates prompt caching at the call site should canonicalize the request so accidental drift doesn't cost cache hits.

The savings are visible in the response's `usage` block: `cache_creation_input_tokens` counts tokens that were just written to cache, and `cache_read_input_tokens` counts tokens served from cache. A library that wraps the call should aggregate these into a structured telemetry value with the dollar amount saved computed against the in-repo pricing table — fabricated savings numbers are worse than no number at all.

## When not to bother

Prompt caching is operationally useful when the prefix is long (≥1024 tokens) and reused enough times to recoup the write cost. A one-shot call with no shared prefix gets nothing from caching. A workflow with a 200-token system prompt and unique user messages every time will see savings that are dominated by accounting overhead.
