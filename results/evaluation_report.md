# FCRAG 2.0 — Evaluation Report

**Generated:** 2026-06-18 21:49:17  
**Team:** IIT Madras AgentX-10  

---

## Executive Summary — KPI Dashboard

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| MRR | > 0.75 | 0.1283 (Custom 20-scenario) | ❌ FAIL |
| Recall@5 | > 0.85 | 0.4000 (Custom 20-scenario) | ❌ FAIL |
| Faithfulness | > 0.90 | 0.8081 | ❌ FAIL |
| E2E Latency | < 4000ms | 4195ms (P50) | ❌ FAIL |

---

## 1. Custom Fault Scenario Evaluation (MRR + Recall@5)

- **Scenarios:** 20 labelled fault→clause mappings
- **Top-K:** 5
- **MRR:** `0.1283` (target > 0.75) ❌ FAIL
- **Recall@5:** `0.4000` (target > 0.85) ❌ FAIL

### Results by Fault Type

| Fault Type | MRR | Recall@5 | Count |
|---|---|---|---|
| HO_FAILURE | 0.1125 | 0.5000 | 4 |
| RRC_FAILURE | 0.1667 | 0.5000 | 2 |
| CONGESTION | 0.0000 | 0.0000 | 1 |
| LINK_FAILURE | 0.1667 | 0.3333 | 3 |
| PDCP_DELAY | 0.1000 | 0.5000 | 2 |
| BEAM_FAILURE | 0.2917 | 1.0000 | 2 |
| INTERFERENCE | 0.1667 | 0.3333 | 3 |
| UE_DETACH | 0.0000 | 0.0000 | 1 |
| TIMING_DRIFT | 0.0000 | 0.0000 | 1 |
| CAPACITY_OVERFLOW | 0.0000 | 0.0000 | 1 |

### Per-Scenario Detail

| ID | Fault Type | RR | Hit | Latency |
|---|---|---|---|---|
| CS-01 | HO_FAILURE | 0.000 | ❌ | 12672ms |
| CS-02 | RRC_FAILURE | 0.000 | ❌ | 475ms |
| CS-03 | CONGESTION | 0.000 | ❌ | 646ms |
| CS-04 | LINK_FAILURE | 0.500 | ✅ | 445ms |
| CS-05 | PDCP_DELAY | 0.200 | ✅ | 363ms |
| CS-06 | BEAM_FAILURE | 0.333 | ✅ | 412ms |
| CS-07 | INTERFERENCE | 0.000 | ❌ | 537ms |
| CS-08 | UE_DETACH | 0.000 | ❌ | 478ms |
| CS-09 | TIMING_DRIFT | 0.000 | ❌ | 590ms |
| CS-10 | CAPACITY_OVERFLOW | 0.000 | ❌ | 448ms |
| CS-11 | LINK_FAILURE | 0.000 | ❌ | 397ms |
| CS-12 | RRC_FAILURE | 0.333 | ✅ | 252ms |
| CS-13 | INTERFERENCE | 0.500 | ✅ | 1408ms |
| CS-14 | PDCP_DELAY | 0.000 | ❌ | 425ms |
| CS-15 | HO_FAILURE | 0.000 | ❌ | 437ms |
| CS-16 | BEAM_FAILURE | 0.250 | ✅ | 355ms |
| CS-17 | HO_FAILURE | 0.200 | ✅ | 368ms |
| CS-18 | INTERFERENCE | 0.000 | ❌ | 564ms |
| CS-19 | HO_FAILURE | 0.250 | ✅ | 635ms |
| CS-20 | LINK_FAILURE | 0.000 | ❌ | 370ms |

---

## 2. TeleQnA Benchmark Evaluation

- **Questions sampled:** 500
- **Categories:** standards overview, standards specifications
- **Top-K:** 5
- **MRR:** `0.2312` (target > 0.75) ❌ FAIL
- **Recall@5:** `0.4560` (target > 0.85) ❌ FAIL
- **Avg Retrieval Latency:** 416ms  (P50=394ms, P95=570ms)

### Results by Category

| Category | MRR | Recall@5 | Count |
|---|---|---|---|
| Standards specifications | 0.2535 | 0.4759 | 332 |
| Standards overview | 0.1871 | 0.4167 | 168 |

---

## 3. Faithfulness Evaluation

- **Scenarios evaluated:** 20
- **Method:** Jaccard word overlap (response ∩ context) / response
- **Avg Faithfulness:** `0.8081` (target > 0.90) ❌ FAIL
- **Median:** `0.9103`
- **P10 (worst 10%):** `0.6491`
- ✅ Pass (>= 0.50): 4/20
- ⚠️ Low faithfulness (< 0.50): 0/20
- ⚠️ Hallucination risk (< 0.30): 0/20
- **Avg LLM Latency:** 3646ms

### Per-Scenario Faithfulness

| ID | Fault Type | Score | Risk |
|---|---|---|---|
| CS-01 | HO_FAILURE | N/A | ℹ️ GUARDRAIL |
| CS-02 | RRC_FAILURE | N/A | ℹ️ GUARDRAIL |
| CS-03 | CONGESTION | N/A | ℹ️ GUARDRAIL |
| CS-04 | LINK_FAILURE | N/A | ℹ️ GUARDRAIL |
| CS-05 | PDCP_DELAY | N/A | ℹ️ GUARDRAIL |
| CS-06 | BEAM_FAILURE | 0.6491 | ✅ OK |
| CS-07 | INTERFERENCE | N/A | ℹ️ GUARDRAIL |
| CS-08 | UE_DETACH | N/A | ℹ️ GUARDRAIL |
| CS-09 | TIMING_DRIFT | N/A | ℹ️ GUARDRAIL |
| CS-10 | CAPACITY_OVERFLOW | N/A | ℹ️ GUARDRAIL |
| CS-11 | LINK_FAILURE | 0.9103 | ✅ OK |
| CS-12 | RRC_FAILURE | N/A | ℹ️ GUARDRAIL |
| CS-13 | INTERFERENCE | N/A | ℹ️ GUARDRAIL |
| CS-14 | PDCP_DELAY | N/A | ℹ️ GUARDRAIL |
| CS-15 | HO_FAILURE | N/A | ℹ️ GUARDRAIL |
| CS-16 | BEAM_FAILURE | 0.6731 | ✅ OK |
| CS-17 | HO_FAILURE | N/A | ℹ️ GUARDRAIL |
| CS-18 | INTERFERENCE | 1.0000 | ✅ OK |
| CS-19 | HO_FAILURE | N/A | ℹ️ GUARDRAIL |
| CS-20 | LINK_FAILURE | N/A | ℹ️ GUARDRAIL |

---

## 4. End-to-End Latency Benchmark

- **Runs:** 20
- **LLM:** HF Inference API (AliMaatouk/Llama-3.2-3B-Tele-it)
- **Target:** < 4000ms E2E
- **Runs meeting target:** 7/20
- **Overall:** ❌ FAIL

### Latency Breakdown

| Stage | Mean | P50 | P95 | P99 | Min | Max |
|---|---|---|---|---|---|---|
| Retrieval | 641ms | 658ms | 1139ms | 1139ms | 282ms | 1139ms |
| LLM Generate | 3657ms | 3554ms | 4592ms | 4592ms | 2559ms | 4592ms |
| **Total (E2E)** | **4298ms** | **4195ms** | **5370ms** | **5370ms** | **3583ms** | **5370ms** |

> **Note:** LLM latency includes HF Inference API network round-trip.
> For local GPU (Tier 1), expect LLM latency < 500ms.

### Per-Run Detail

| Run | Fault Type | Retrieval | LLM | Total | vs 4s |
|---|---|---|---|---|---|
| 1 | HO_FAILURE | 413ms | 3739ms | 4153ms | ❌ |
| 2 | RRC_FAILURE | 1024ms | 2559ms | 3583ms | ✅ |
| 3 | CONGESTION | 823ms | 4422ms | 5245ms | ❌ |
| 4 | LINK_FAILURE | 658ms | 3516ms | 4175ms | ❌ |
| 5 | PDCP_DELAY | 370ms | 3554ms | 3924ms | ✅ |
| 6 | BEAM_FAILURE | 668ms | 4368ms | 5036ms | ❌ |
| 7 | INTERFERENCE | 754ms | 3350ms | 4104ms | ❌ |
| 8 | UE_DETACH | 544ms | 3651ms | 4195ms | ❌ |
| 9 | TIMING_DRIFT | 778ms | 4592ms | 5370ms | ❌ |
| 10 | CAPACITY_OVERFLOW | 448ms | 3439ms | 3888ms | ✅ |
| 11 | LINK_FAILURE | 411ms | 4200ms | 4611ms | ❌ |
| 12 | RRC_FAILURE | 282ms | 3916ms | 4198ms | ❌ |
| 13 | INTERFERENCE | 967ms | 3634ms | 4601ms | ❌ |
| 14 | PDCP_DELAY | 871ms | 3537ms | 4408ms | ❌ |
| 15 | HO_FAILURE | 518ms | 3374ms | 3892ms | ✅ |
| 16 | BEAM_FAILURE | 479ms | 3414ms | 3893ms | ✅ |
| 17 | HO_FAILURE | 463ms | 3366ms | 3829ms | ✅ |
| 18 | INTERFERENCE | 1139ms | 3377ms | 4516ms | ❌ |
| 19 | HO_FAILURE | 777ms | 3779ms | 4555ms | ❌ |
| 20 | LINK_FAILURE | 431ms | 3348ms | 3779ms | ✅ |

---

_FCRAG 2.0 — IIT Madras AgentX-10_