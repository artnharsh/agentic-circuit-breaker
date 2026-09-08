# Circuit Breaker Pattern: Resilience Engineering

## Origin: Netflix Hystrix

The circuit breaker pattern was popularized by Netflix through their Hystrix library
(released 2012, maintained until ~2018). Hystrix was designed to handle failures in
distributed microservices by detecting when a downstream service was failing and
"opening the circuit" to fail fast instead of waiting for timeouts.

## States of the Circuit Breaker

A standard circuit breaker has three states:

1. **CLOSED (normal operation)**: Requests pass through normally. The breaker tracks
   failures. If the failure rate exceeds a threshold, it transitions to OPEN.

2. **OPEN (circuit tripped)**: Requests are rejected immediately without being sent to
   the failing service. A timer runs to allow the service time to recover.

3. **HALF-OPEN (testing recovery)**: After the timer expires, a limited number of test
   requests are allowed through. If they succeed, the breaker returns to CLOSED.
   If they fail, it returns to OPEN.

## Traditional vs. Semantic Circuit Breaker

Traditional circuit breakers (Hystrix-style) use numeric signals:
- Error rate exceeds threshold (e.g., >50% failures in 10 seconds)
- Response latency exceeds threshold (e.g., >2 seconds)

The Agentic Circuit Breaker uses a **semantic signal** instead:
- Cosine similarity between consecutive agent reasoning attempts
- High similarity (≥0.85) indicates the agent is repeating itself (thrashing)
- This semantic signal is LLM-native and captures meaning, not just errors/latency

## Why This Is Novel

No existing system combines:
1. A stateful CLOSED → HALF-OPEN → OPEN finite state machine
2. An LLM-native semantic signal (embedding cosine similarity)
3. Real-time, mid-run intervention (not post-hoc analysis)

This combination is the core research contribution of this project.
