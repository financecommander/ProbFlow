# Swarm Data Governance — ProbFlow

**Calculus Ecosystem • Governance Rule 4 Compliance**

## Rule

> **super-duper-spork retains ALL swarm management data.**
>
> External repos that generate, test, or benchmark swarm algorithms
> must push assessment results back to `financecommander/super-duper-spork`.

## This Repo's Obligations

ProbFlow owns probabilistic task scheduling and confidence scoring for
the swarm. When scheduling models produce swarm routing weights, algorithm
selection probabilities, or worker reliability metrics, those must be reported.

| Obligation | Target | Format |
|-----------|--------|--------|
| Push probabilistic routing/scheduling metrics | `super-duper-spork/swarm/assessments/` | `probflow_{description}_{YYYY-MM-DD}.md` |
| Report algorithm confidence calibration data | `super-duper-spork/swarm/assessments/` | Assessment markdown |

## Canonical Source of Truth

The single source of truth for all swarm state is:

    financecommander/super-duper-spork

This repo handles probabilistic scheduling. super-duper-spork retains all
swarm management data, assessment results, and cross-repo totals.

---
*Governance Rule 4 — established 2026-03-09*
