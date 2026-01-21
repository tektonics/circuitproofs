# MBPP-Lean Benchmark Integration

This module integrates the MBPP (Mostly Basic Python Problems) subset from the [VERINA benchmark](https://github.com/sunblaze-ucb/verina), which provides 49 MBPP problems manually translated to Lean 4 with formal specifications.

## Purpose

MBPP problems with Lean specifications allow us to:

1. **Validate Circuit Extraction**: Extract circuits from models solving MBPP problems, then verify the circuits match the formal Lean specifications
2. **Enable Counterfactual Testing**: Test that circuits capture semantics, not syntax, by comparing circuits from input variants (renamed variables, reordered arguments)
3. **Provide Ground Truth**: The pre-written Lean specifications serve as the "answer key" for what extracted circuits should compute

## How It Works

```
Model solves MBPP problem (e.g., "calculate average")
         ↓
Extract circuit from model
         ↓
Translate circuit to Lean
         ↓
Prove circuit matches MBPP Lean specification
         ↓
Certificate: Circuit correctly implements the algorithm
```

## Dataset Source

The MBPP-Lean specifications come from VERINA-BASIC, which contains:
- 49 MBPP problems translated from MBPP-DFY-50
- Each problem includes: natural language description, Lean implementation, formal specification, proof, and test cases
- Human-verified translations (not LLM-generated)

## Setup

```bash
# Fetch the MBPP-Lean dataset
python benchmarks/verina/fetch_dataset.py

# Run benchmarks on MBPP subset
python benchmarks/verina/run_benchmark.py --subset mbpp
```

## Directory Structure

```
benchmarks/verina/
├── README.md              # This file
├── fetch_dataset.py       # Download VERINA dataset, extract MBPP subset
├── run_benchmark.py       # Execute benchmark pipeline
├── variant_generator.py   # Generate input variants for counterfactual testing
├── circuit_comparator.py  # Compare circuits for structural equivalence
└── data/                  # Downloaded tasks (gitignored)
    └── mbpp/              # MBPP-specific problems
```

## Counterfactual Testing

To verify that circuit extraction captures true computational semantics (not surface patterns):

1. Take an MBPP problem (e.g., "calculate average of a list")
2. Generate variants:
   - Rename variables: `numbers` → `values`
   - Reorder parameters: `(a, b)` → `(b, a)`
   - Change formatting: whitespace, comments
3. Have model solve each variant
4. Extract circuits from each solution
5. Verify circuits are structurally equivalent

If circuits differ significantly for semantically identical problems, the model has shallow understanding.

## Integration with circuitproofs

```python
from benchmarks.verina import load_mbpp_task, verify_circuit_against_spec

# Load an MBPP task with its Lean specification
task = load_mbpp_task("task_001")

# Extract circuit from your model's solution
circuit = extract_circuit(model, task.input)

# Verify circuit matches the formal specification
result = verify_circuit_against_spec(circuit, task.lean_spec)
```

## References

- [VERINA Paper](https://arxiv.org/abs/2505.23135)
- [VERINA GitHub](https://github.com/sunblaze-ucb/verina)
- [VERINA Dataset](https://huggingface.co/datasets/sunblaze-ucb/verina)
- [Original MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
