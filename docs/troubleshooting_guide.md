# API Testing & CI/CD Troubleshooting Guide

This document addresses common issues with API testing and CI/CD pipeline configuration that were resolved during project preparation for public release.

## Issues Resolved

### 1. API Test 503 Service Unavailable Errors

**Problem**: All API tests were failing with 503 Service Unavailable errors instead of expected HTTP status codes.

**Root Cause**: Relative import issues in the API module. The FastAPI application was using absolute imports (`from config import...`) which failed when the package was properly installed.

**Solution**: Updated all imports in `src/` modules to use relative imports:

```python
# Before (causing 503 errors)
from config import BEST_MODEL_FILE, CATEGORICAL_COLUMNS
from predict import predict, validate_output_schema
from utils import load_pickle

# After (fixed)
from .config import BEST_MODEL_FILE, CATEGORICAL_COLUMNS
from .predict import predict, validate_output_schema
from .utils import load_pickle
```

**Files Fixed**:
- `src/api.py` - Main FastAPI application
- `src/predict.py` - Prediction logic
- `src/preprocess.py` - Data preprocessing
- `src/train.py` - Model training
- `src/run_pipeline.py` - Pipeline orchestration

### 2. CI/CD Dependency Installation Misalignment

**Problem**: GitHub Actions CI was trying to install from `requirements.txt` but the project uses `pyproject.toml` for dependency management.

**Root Cause**: Outdated CI configuration referencing non-existent `requirements.txt`.

**Solution**: Updated `.github/workflows/ci.yml` to use modern Python packaging:

```yaml
# Before
pip install -r requirements.txt

# After  
pip install -e .[dev]
```

This installs the package in development mode with test dependencies as defined in `pyproject.toml`.

## Verification Steps

### Testing API Fixes

1. **Install the package**:
   ```bash
   pip install -e .[dev]
   ```

2. **Run API tests**:
   ```bash
   pytest tests/test_api.py -v
   ```

3. **Expected results**: All tests should pass with proper HTTP status codes (200, 400, 500) instead of 503.

### Testing CI/CD Fixes

1. **Push changes to trigger CI**:
   ```bash
   git add .
   git commit -m "Fix API imports and CI dependency installation"
   git push
   ```

2. **Check GitHub Actions**: Verify that the workflow runs without dependency installation errors.

## Prevention Measures

### Import Best Practices

- Always use relative imports within packages (`from .module import ...`)
- Use absolute imports only for external packages
- Test imports in both development and installed environments

### CI/CD Configuration

- Keep CI configuration aligned with project's dependency management
- Use `pyproject.toml` as the single source of truth for dependencies
- Test CI configuration locally before pushing

### Testing Strategy

- Run tests both in development and after package installation
- Include import testing in the test suite
- Test API endpoints with proper HTTP status validation

## Common Error Messages and Solutions

### ModuleNotFoundError: No module named 'config'

**Cause**: Absolute imports in package modules
**Solution**: Use relative imports (`from .config import ...`)

### pip install -r requirements.txt fails

**Cause**: CI misaligned with project structure
**Solution**: Update CI to use `pip install -e .[dev]`

### 503 Service Unavailable in API tests

**Cause**: Import failures preventing application startup
**Solution**: Fix import statements and verify package installation

## Additional Resources

- [Python Packaging Documentation](https://packaging.python.org/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## Maintenance

- Review import statements when adding new modules
- Keep CI configuration updated with dependency changes
- Run full test suite before releases
- Monitor CI pipeline for new issues
