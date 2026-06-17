# Telco Churn Analytics Pipeline
# Execute notebooks in dependency order

.PHONY: help install pipeline clean check-deps

# Default target
help:
	@echo "Telco Churn Analytics Pipeline"
	@echo "=============================="
	@echo ""
	@echo "Available commands:"
	@echo "  make help          Show this help"
	@echo "  make install       Install pipeline dependencies"
	@echo "  make check-deps    Check if all dependencies are met"
	@echo "  make pipeline      Run complete pipeline"
	@echo "  make pipeline-3    Start from notebook 03 (feature engineering)"
	@echo "  make pipeline-4    Start from notebook 04 (modeling)"
	@echo "  make clean         Clean generated files"
	@echo ""
	@echo "Pipeline order:"
	@echo "  1. 01_eda.ipynb"
	@echo "  2. 02_preprocessing.ipynb"
	@echo "  3. 03_feature_engineering.ipynb"
	@echo "  4. 04_modeling.ipynb"
	@echo "  5. 05_results.ipynb"

# Install dependencies
install:
	@echo "📦 Installing pipeline dependencies..."
	pip install -e ".[pipeline]"
	pip install -e ".[interactive]"
	@echo "✅ Installation complete"

# Check dependencies
check-deps:
	@echo "🔍 Checking dependencies..."
	@python scripts/run_pipeline.py --dry-run

# Run complete pipeline
pipeline:
	@echo "🚀 Running complete pipeline..."
	@python scripts/run_pipeline.py

# Start from notebook 03
pipeline-3:
	@echo "🚀 Starting from notebook 03 (feature engineering)..."
	@python scripts/run_pipeline.py --start 3

# Start from notebook 4
pipeline-4:
	@echo "🚀 Starting from notebook 04 (modeling)..."
	@python scripts/run_pipeline.py --start 4

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@find . -name "*.executed.ipynb" -delete
	@rm -rf data/processed/*.parquet
	@rm -rf models/*.pkl
	@rm -rf results/*.html
	@rm -rf results/*.png
	@echo "✅ Clean complete"

# Quick status check
status:
	@echo "📊 Pipeline Status"
	@echo "=================="
	@echo "Data files:"
	@ls -la data/processed/ 2>/dev/null || echo "  No processed data files"
	@echo ""
	@echo "Model files:"
	@ls -la models/ 2>/dev/null || echo "  No model files"
	@echo ""
	@echo "Result files:"
	@ls -la results/ 2>/dev/null || echo "  No result files"
