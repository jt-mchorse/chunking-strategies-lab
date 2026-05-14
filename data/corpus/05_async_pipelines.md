# Async LLM Pipelines: Patterns That Actually Help

## The default is too slow

A pipeline that calls an LLM once per document, serially, on a 10,000-document workload is bottlenecked entirely on network latency. Each call waits a few hundred milliseconds to a few seconds; multiply by 10,000 and you have a multi-hour run that should have been minutes. Replacing the serial loop with structured concurrency is one of the highest-leverage refactors in any LLM-heavy codebase.

## Async batching with bounded concurrency

The simplest improvement that survives contact with production rate limits is async batching with a `Semaphore`-bounded fan-out:

```python
sem = asyncio.Semaphore(MAX_CONCURRENT)

async def process_one(doc):
    async with sem:
        return await llm.acomplete(doc)

results = await asyncio.gather(*(process_one(d) for d in docs))
```

Tune `MAX_CONCURRENT` to your rate-limit quota minus a margin. Going past the rate limit costs more than it saves because the requests get queued by the provider — and a queued request is no faster than a serial one.

## Concurrent tool-call dispatch

Agent workflows often issue several tool calls per turn. The default in many SDKs is to dispatch them serially, which leaves latency on the floor. When the tools are independent, dispatch them concurrently:

```python
calls = response.tool_calls
results = await asyncio.gather(*(dispatch(c) for c in calls))
```

The savings are proportional to the number of independent calls. For a research agent that runs four searches per turn, this is a 4× per-turn speedup before any other optimization.

## Backpressure handling

When the consumer of an LLM pipeline is slower than the producer, an unbounded queue grows until the process runs out of memory. The standard mitigation is a bounded `asyncio.Queue`: producers block on `put` when the queue is full, which propagates backpressure upstream.

For pipelines that can't block — e.g., webhook receivers — the alternative is shed-load: drop or 429 incoming requests when the queue exceeds a threshold. This is operationally clearer than a memory blow-up and easier to monitor.

## Structured concurrency with TaskGroup

Python 3.11 added `asyncio.TaskGroup`, which gives you structured concurrency primitives that propagate exceptions cleanly. Prefer it over `asyncio.gather` for new code:

```python
async with asyncio.TaskGroup() as tg:
    tasks = [tg.create_task(process_one(d)) for d in docs]
results = [t.result() for t in tasks]
```

The difference matters when a downstream task raises: `gather(return_exceptions=False)` cancels everything but doesn't always clean up tidily; `TaskGroup` always does, and it surfaces the exception type at the point of the `async with`.

## Benchmarks worth running

For any of these patterns, the relevant benchmarks are:

- Serial baseline (the unoptimized version).
- Async-bounded (semaphore at provider rate limit).
- Async-batched (provider's batch endpoint when available).

Expect a 5–20× speedup from serial → async-bounded on a workload with realistic per-call latency. Going beyond that usually requires the provider's native batch endpoint, which trades latency for throughput and isn't right for every use case.
