"""
Fetch VERINA Dataset and Extract MBPP Subset

Downloads the VERINA benchmark from HuggingFace and extracts
the MBPP problems that have been translated to Lean 4.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MBPPTask:
    """Represents an MBPP task with Lean specification."""

    task_id: str
    description: str
    lean_code: str
    lean_spec: str
    lean_proof: str
    test_cases: List[Dict]
    source: str  # "mbpp" for MBPP-derived tasks


def fetch_verina_dataset(
    output_dir: Path,
    subset: str = "mbpp"
) -> List[MBPPTask]:
    """
    Fetch VERINA dataset from HuggingFace.

    Args:
        output_dir: Directory to save downloaded data
        subset: Which subset to fetch ("mbpp", "basic", "advanced", "all")

    Returns:
        List of MBPPTask objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the datasets library: pip install datasets"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching VERINA dataset from HuggingFace...")
    dataset = load_dataset("sunblaze-ucb/verina")

    tasks = []
    mbpp_count = 0

    # Process the dataset
    for split_name in dataset.keys():
        split_data = dataset[split_name]

        for item in split_data:
            # Check if this is an MBPP-derived task
            # VERINA-BASIC contains MBPP problems
            task_id = item.get("id", item.get("task_id", "unknown"))

            # Filter based on subset
            is_mbpp = _is_mbpp_task(item)

            if subset == "mbpp" and not is_mbpp:
                continue
            elif subset == "basic" and not _is_basic_task(item):
                continue
            elif subset == "advanced" and not _is_advanced_task(item):
                continue

            task = MBPPTask(
                task_id=task_id,
                description=item.get("description", ""),
                lean_code=item.get("code", item.get("lean_code", "")),
                lean_spec=item.get("specification", item.get("spec", "")),
                lean_proof=item.get("proof", ""),
                test_cases=item.get("tests", item.get("test_cases", [])),
                source="mbpp" if is_mbpp else "other"
            )

            tasks.append(task)

            if is_mbpp:
                mbpp_count += 1

            # Save individual task
            task_dir = output_dir / task_id
            task_dir.mkdir(exist_ok=True)

            _save_task(task, task_dir)

    print(f"Fetched {len(tasks)} tasks ({mbpp_count} MBPP-derived)")

    # Save manifest
    manifest = {
        "total_tasks": len(tasks),
        "mbpp_tasks": mbpp_count,
        "subset": subset,
        "task_ids": [t.task_id for t in tasks]
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return tasks


def _is_mbpp_task(item: Dict) -> bool:
    """Check if a task is derived from MBPP."""
    # VERINA-BASIC contains MBPP translations
    task_id = str(item.get("id", item.get("task_id", ""))).lower()
    source = str(item.get("source", "")).lower()

    return "mbpp" in source or "mbpp" in task_id or item.get("is_mbpp", False)


def _is_basic_task(item: Dict) -> bool:
    """Check if task is from VERINA-BASIC subset."""
    return item.get("subset", "").lower() == "basic"


def _is_advanced_task(item: Dict) -> bool:
    """Check if task is from VERINA-ADV subset."""
    return item.get("subset", "").lower() in ("advanced", "adv")


def _save_task(task: MBPPTask, task_dir: Path) -> None:
    """Save a task to disk."""
    # Save metadata
    metadata = {
        "task_id": task.task_id,
        "description": task.description,
        "source": task.source,
        "test_cases": task.test_cases
    }

    with open(task_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save Lean files
    if task.lean_code:
        with open(task_dir / "code.lean", "w") as f:
            f.write(task.lean_code)

    if task.lean_spec:
        with open(task_dir / "spec.lean", "w") as f:
            f.write(task.lean_spec)

    if task.lean_proof:
        with open(task_dir / "proof.lean", "w") as f:
            f.write(task.lean_proof)


def load_mbpp_task(task_id: str, data_dir: Optional[Path] = None) -> MBPPTask:
    """
    Load a single MBPP task from disk.

    Args:
        task_id: The task identifier
        data_dir: Directory containing downloaded tasks

    Returns:
        MBPPTask object
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    task_dir = data_dir / task_id

    if not task_dir.exists():
        raise FileNotFoundError(f"Task {task_id} not found in {data_dir}")

    with open(task_dir / "metadata.json") as f:
        metadata = json.load(f)

    lean_code = ""
    lean_spec = ""
    lean_proof = ""

    if (task_dir / "code.lean").exists():
        with open(task_dir / "code.lean") as f:
            lean_code = f.read()

    if (task_dir / "spec.lean").exists():
        with open(task_dir / "spec.lean") as f:
            lean_spec = f.read()

    if (task_dir / "proof.lean").exists():
        with open(task_dir / "proof.lean") as f:
            lean_proof = f.read()

    return MBPPTask(
        task_id=metadata["task_id"],
        description=metadata["description"],
        lean_code=lean_code,
        lean_spec=lean_spec,
        lean_proof=lean_proof,
        test_cases=metadata.get("test_cases", []),
        source=metadata.get("source", "unknown")
    )


def list_available_tasks(data_dir: Optional[Path] = None) -> List[str]:
    """List all available task IDs."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    if not data_dir.exists():
        return []

    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        return manifest.get("task_ids", [])

    # Fallback: list directories
    return [d.name for d in data_dir.iterdir() if d.is_dir()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch VERINA/MBPP dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--subset",
        choices=["mbpp", "basic", "advanced", "all"],
        default="mbpp",
        help="Which subset to fetch"
    )

    args = parser.parse_args()

    tasks = fetch_verina_dataset(args.output_dir, args.subset)
    print(f"\nDownloaded {len(tasks)} tasks to {args.output_dir}")
