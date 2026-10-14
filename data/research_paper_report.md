# Agentic Circuit Breaker: Empirical Evaluation Report

**Date:** 2026-10-14 14:29:27 UTC  
**Sample Size:** 3 queries evaluated  

---

## Executive Summary

The Agentic Circuit Breaker successfully identified and interrupted semantic thrashing in **100.0%** of adversarial evaluation queries.
Across all evaluation queries, the circuit breaker reduced token consumption by an average of **56.5%** (5,366 tokens saved per run) while averting unhandled system crashes (`GraphRecursionError`).

---

## Table 1: Baseline vs Circuit Breaker Performance Comparison

| Metric | Baseline (Unprotected) | Circuit Breaker (Protected) | Relative Delta |
|:---|:---:|:---:|:---:|
| **Mean Token Consumption** | 9,497.0 ± 23.6 | 4,131.0 ± 10.8 | **-56.5%** |
| **Mean Iterations Completed** | 15.0 | 2.0 | -86.7% |
| **Failure / Crash Rate** | 100.0% (`GraphRecursionError`) | 0.0% (Graceful partial synthesis) | **-100.0%** |
| **Intervention Rate** | 0.0% | 100.0% | — |

---

## Table 2: Inferential Statistics & Effect Size

| Statistical Test | Parameter / Statistic | Value | Interpretation |
|:---|:---:|:---:|:---|
| **Paired Samples t-test** | $t(2)$ | 727.9767 | p < 0.001 (Highly Significant) |
| **Significance (p-value)** | $p$ | 0.000002 | Rejects null hypothesis ($H_0$) |
| **Cohen's $d$ Effect Size** | $d$ | 420.2976 | Huge effect ($d > 0.8$) |

---

## Table 3: Per-Node Fault Localization Analysis

This table demonstrates the **per-node observability** capability, enabling developers to pinpoint which agent in the multi-agent pipeline is generating redundant outputs.

| Faulty Agent Identified | Incident Count | Share of Thrashing Incidents | Typical Fault Mode |
|:---|:---:|:---:|:---|
| `researcher` | 3 | 100.0% | Repeated extraction attempts of unavailable corpus data |

---

## Detailed Query Log

| # | Query Snippet | Baseline Status (Tokens) | Protected Status (Tokens) | Tripped Agent | Saved Tokens (%) |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | What is the exact CO2 reduction achieved per count... | crashed (9,523) | breaker_triggered (4,143) | `researcher` | 5,380 (56.5%) |
| 2 | Provide the exact solar panel efficiency improveme... | crashed (9,477) | breaker_triggered (4,122) | `researcher` | 5,355 (56.5%) |
| 3 | What were the precise GDP impact coefficients for ... | crashed (9,491) | breaker_triggered (4,128) | `researcher` | 5,363 (56.5%) |
