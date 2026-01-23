# Proof Completion Roadmap

This document tracks the `sorry` placeholders in the Lean codebase and prioritizes which proofs should be completed first for maximum impact.

## Priority Levels

- **P0 (Critical)**: Core theorems needed to demonstrate the pipeline works end-to-end
- **P1 (High)**: Important for grant applications and demonstrating value
- **P2 (Medium)**: Strengthens the formal guarantees
- **P3 (Low)**: Nice to have, can be deferred

---

## P0: Critical Path Proofs

These proofs are essential for demonstrating the circuit verification pipeline works.

### Circuit Property Transfer
**File**: `lean/FormalVerifML/base/circuit_models.lean:217`
```lean
theorem property_transfer (circuit : Circuit)
    (originalModel : Array Float → Array Float)
    (property : Array Float → Prop)
    (propertyLipschitz : Float) :
  circuitSatisfiesProperty circuit property 1.0 →
  circuitApproximatesModel circuit originalModel →
  circuit.errorBound.epsilon < propertyLipschitz →
  (∀ x, property (originalModel x))
```
**Why Critical**: This is the core theorem—it proves that properties verified on the circuit transfer to the original model. Without this, the whole pipeline's value proposition is unproven.

### Lipschitz Composition Bound
**File**: `lean/FormalVerifML/base/circuit_models.lean:203`
```lean
theorem lipschitz_composition_bound (circuit : Circuit) : ...
```
**Why Critical**: This justifies the error bound computation used throughout the pipeline.

---

## P1: High Priority Proofs

Important for demonstrating specific verification capabilities.

### Circuit Robustness
**File**: `lean/FormalVerifML/proofs/circuit_proofs.lean:48`
```lean
theorem circuit_robustness_example (circuit : Circuit) (δ ε : Float) : ...
```
**Why Important**: Robustness is the most common property people want to verify.

### Circuit Monotonicity
**File**: `lean/FormalVerifML/proofs/circuit_proofs.lean:91`
```lean
theorem circuit_monotonic_example (circuit : Circuit) (featureIdx : Nat) : ...
```
**Why Important**: Monotonicity is key for fairness verification (e.g., "higher credit score should not decrease loan approval probability").

### Complete Circuit Verification
**File**: `lean/FormalVerifML/proofs/circuit_proofs.lean:248`
```lean
theorem complete_circuit_verification ...
```
**Why Important**: This is the showcase theorem combining all verification steps.

---

## P2: Medium Priority Proofs

### Circuit Equivalence (Counterfactual Testing)
**File**: `lean/FormalVerifML/base/circuit_equivalence.lean:42-53`
```lean
theorem structural_implies_semantic ...
theorem structural_equiv_refl ...
theorem structural_equiv_symm ...
```
**Why Important**: Enables the counterfactual testing workflow described in the MBPP-Lean integration.

### Fairness Proofs
**File**: `lean/FormalVerifML/proofs/example_fairness_proof.lean:53`
```lean
theorem example_demographic_parity ...
```
**Why Important**: Fairness is a key selling point for regulatory applications.

### Error Bound Validity
**File**: `lean/FormalVerifML/proofs/circuit_proofs.lean:138`
```lean
theorem error_bound_valid (circuit : Circuit) : ...
```

---

## P3: Lower Priority Proofs

### Sparsity Computation
**File**: `lean/FormalVerifML/proofs/circuit_proofs.lean:32`
```lean
theorem simpleLinearCircuit_sparse : circuitSparsity simpleLinearCircuit > 0
```

### Decision Tree Verification
**File**: `lean/FormalVerifML/proofs/decision_tree_proof.lean:26`

### Extended Robustness
**File**: `lean/FormalVerifML/proofs/extended_robustness_proof.lean:25`

### Interpretability
**File**: `lean/FormalVerifML/proofs/circuit_proofs.lean:222`

---

## Summary

| Priority | Count | Focus Area |
|----------|-------|------------|
| P0 | 2 | Core pipeline theorems |
| P1 | 3 | Key verification properties |
| P2 | 4 | Supporting guarantees |
| P3 | 4+ | Nice-to-have |

**Recommended approach**: Complete P0 proofs first to have a working end-to-end demonstration, then P1 for grant application strength.

---

## How to Work on Proofs

1. Start with the proof strategy comments in each theorem
2. Use `#check` and `#eval` to explore types and values
3. Break complex proofs into smaller lemmas
4. Reference mathlib for existing theorems about reals, arrays, etc.

See [Lean 4 documentation](https://leanprover.github.io/lean4/doc/) and [mathlib4 docs](https://leanprover-community.github.io/mathlib4_docs/) for reference.
