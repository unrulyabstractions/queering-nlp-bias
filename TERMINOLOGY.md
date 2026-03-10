# Terminology

Core concepts from structure-aware diversity theory.

## Structure & System

A **structure** α scores how well a string exhibits a behavior of interest:
```
α: String → [0, 1]     (e.g., "Does this mention a woman?")
```

A **system** Λ is a vector of multiple structure compliances:
```
Λ_n(x) = (α_1(x), α_2(x), ..., α_n(x))
```

## Core

The **core** ⟨Λ_n⟩ is the expected system compliance over all possible continuations:
```
⟨Λ_n⟩ = Σ p(y) · Λ_n(y)     weighted by trajectory probability
```

**Interpretation**: The core describes what the model treats as *default* for each structure. It encodes normativity.

## Orientation

The **orientation** θ measures how a string deviates from the core:
```
θ_n(x) = Λ_n(x) - ⟨Λ_n⟩     (vector of per-structure deviations)
```

**Interpretation**: Positive values = above-normal compliance. Negative = below-normal.

## Deviance

The **deviance** ∂ collapses orientation to a single scalar (L2 norm):
```
∂_n(x) = ||θ_n(x)||         (total deviation from normativity)
```

**Interpretation**: High deviance = non-normative. Low deviance = normative (close to the core).

## Summary

| Concept     | Symbol | Type   | Meaning |
|-------------|--------|--------|---------|
| Compliance  | α(x)   | scalar | How well x matches one structure |
| System      | Λ_n(x) | vector | Compliance across all structures |
| Core        | ⟨Λ_n⟩  | vector | Expected compliance (the "normal") |
| Orientation | θ_n(x) | vector | Per-structure deviation from core |
| Deviance    | ∂_n(x) | scalar | Total non-normativity |

## Reference

Based on "Structure-Aware Diversity Pursuit as an AI Safety Strategy against Homogenization" (ACL submission).
