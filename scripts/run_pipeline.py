#!/usr/bin/env python3
"""
Automated Pipeline Runner

Executes all Jupyter notebooks in sequence to create a complete ML pipeline.
Handles dependencies, error handling, and progress tracking.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --start 3  # Start from notebook 03
    python scripts/run_pipeline.py --dry-run  # Show execution plan
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Notebook execution order
NOTEBOOK_ORDER = [
    "01_eda.ipynb",
    "02_preprocessing.ipynb",
    "03_feature_engineering.ipynb",
    "04_modeling.ipynb",
    "05_results.ipynb"
]

# Required files for each notebook
DEPENDENCIES = {
    "01_eda.ipynb": ["data/raw/telco_customer_churn.csv"],
    "02_preprocessing.ipynb": ["data/raw/telco_customer_churn.csv"],
    "03_feature_engineering.ipynb": [
        "data/processed/train.parquet",
        "data/processed/test.parquet",
        "data/processed/train_labels.parquet",
        "data/processed/test_labels.parquet",
        "data/raw/telco_customer_churn.csv"
    ],
    "04_modeling.ipynb": [
        "data/processed/train_engineered.parquet",
        "data/processed/test_engineered.parquet",
        "data/processed/train_labels.parquet",
        "data/processed/test_labels.parquet"
    ],
    "05_results.ipynb": [
        "data/processed/test_engineered.parquet",
        "data/processed/test_labels.parquet",
        "models/gradient_boosting.pkl"
    ]
}


def check_dependencies(notebook: str, notebooks_dir: Path) -> bool:
    """Check if all required files exist for a notebook."""
    project_root = notebooks_dir.parent
    missing_files = []

    for dep_path in DEPENDENCIES.get(notebook, []):
        full_path = project_root / dep_path
        if not full_path.exists():
            missing_files.append(str(full_path))

    if missing_files:
        print(f"❌ Missing dependencies for {notebook}:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    return True


def run_notebook(notebook_path: Path, timeout: int = 600) -> tuple[bool, str]:
    """Execute a single Jupyter notebook."""
    print(f"\n🚀 Running: {notebook_path.name}")
    print(f"   Path: {notebook_path}")

    try:
        # Use papermill for reliable notebook execution
        cmd = [
            sys.executable, "-m", "papermill",
            str(notebook_path),
            str(notebook_path.with_suffix('.executed.ipynb')),
            "--kernel", "python3",
            "--execution-timeout", str(timeout)
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=notebook_path.parent
        )
        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Completed in {execution_time:.1f}s")
            return True, f"Success ({execution_time:.1f}s)"
        else:
            print(f"❌ Failed after {execution_time:.1f}s")
            print(f"Error: {result.stderr}")
            return False, f"Failed: {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def run_pipeline(
    start_notebook: int = 1,
    dry_run: bool = False,
    timeout: int = 600
) -> bool:
    """Run the complete notebook pipeline."""

    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    project_root = notebooks_dir.parent

    print("🎯 Telco Churn Analytics Pipeline")
    print("=" * 50)
    print(f"Notebooks directory: {notebooks_dir}")
    print(f"Project root: {project_root}")

    # Filter notebooks based on start position
    notebooks_to_run = NOTEBOOK_ORDER[start_notebook - 1:]

    print(f"\n📋 Execution Plan:")
    for i, notebook in enumerate(notebooks_to_run, start=start_notebook):
        status = "🔄" if i == start_notebook else "⏸️"
        print(f"   {status} {i:02d}. {notebook}")

    if dry_run:
        print("\n🔍 Dry run mode - no execution")
        return True

    print(f"\n⚡ Starting pipeline from notebook {start_notebook}")
    print("=" * 50)

    # Track execution results
    results = []
    successful = True

    for i, notebook in enumerate(notebooks_to_run, start=start_notebook):
        notebook_path = notebooks_dir / notebook

        # Check if notebook exists
        if not notebook_path.exists():
            print(f"❌ Notebook not found: {notebook_path}")
            results.append((i, notebook, "Not found", False))
            successful = False
            continue

        # Check dependencies
        if not check_dependencies(notebook, notebooks_dir):
            print(f"❌ Skipping {notebook} due to missing dependencies")
            results.append((i, notebook, "Missing dependencies", False))
            successful = False
            continue

        # Execute notebook
        success, message = run_notebook(notebook_path, timeout)
        results.append((i, notebook, message, success))

        if not success:
            print(f"\n🛑 Pipeline stopped at notebook {i}")
            successful = False
            break

    # Print summary
    print("\n" + "=" * 50)
    print("📊 Pipeline Summary")
    print("=" * 50)

    for i, notebook, message, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {i:02d}. {notebook:<25} {message}")

    if successful:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Generated files:")

        # List generated files
        data_dir = project_root / "data"
        models_dir = project_root / "models"
        results_dir = project_root / "results"

        for directory in [data_dir, models_dir, results_dir]:
            if directory.exists():
                files = list(directory.rglob("*"))
                print(f"   {directory.name}: {len(files)} files")
    else:
        print(f"\n❌ Pipeline failed. Check error messages above.")

    return successful


def install_papermill():
    """Install papermill if not available."""
    try:
        import papermill
        return True
    except ImportError:
        print("📦 Installing papermill for reliable notebook execution...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "papermill"
        ], check=True)
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run telco churn analytics pipeline")
    parser.add_argument("--start", type=int, default=1,
                       help="Starting notebook number (1-5)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show execution plan without running")
    parser.add_argument("--timeout", type=int, default=600,
                       help="Timeout per notebook in seconds")

    args = parser.parse_args()

    # Validate start notebook
    if not 1 <= args.start <= len(NOTEBOOK_ORDER):
        print(f"❌ Start notebook must be between 1 and {len(NOTEBOOK_ORDER)}")
        return 1

    # Install papermill
    if not args.dry_run:
        if not install_papermill():
            print("❌ Failed to install papermill")
            return 1

    # Run pipeline
    success = run_pipeline(
        start_notebook=args.start,
        dry_run=args.dry_run,
        timeout=args.timeout
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
