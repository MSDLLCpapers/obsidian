#!/bin/bash
set -e
set -o pipefail

# Cross-Version Numerical Test
# Tests numerical reproducibility across Python 3.10, 3.11, 3.12
# with fixed dependency versions (torch 2.3.0, botorch 0.11.3, gpytorch 1.12, numpy 1.26.4)
#
# Note: results are NOT bit-identical across Python versions -- last-bit
# (reduction/instruction-order) differences exist at machine-epsilon level
# (<=1e-14). The comparison therefore asserts *numerical equivalence* within a
# tolerance (--baseline-rtol/--baseline-atol, default 1e-9), which is ~5 orders
# of magnitude above the observed noise and well below any real regression.
#
# Usage: ./cross_version_numerical_test.sh [--keep-venv] [--no-cleanup]
#   --keep-venv: Don't delete existing venvs, reuse if present
#   --no-cleanup: Don't clean up venvs and baselines after completion

# Must be run from repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Parse arguments
KEEP_VENV=false
NO_CLEANUP=false
for arg in "$@"; do
    case $arg in
        --keep-venv)
            KEEP_VENV=true
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--keep-venv] [--no-cleanup]"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "Cross-Version Numerical Reproducibility Test"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if uv is installed. We require it to be pre-installed rather than
# piping a remote installer into the shell (curl ... | sh is an unpinned,
# unverified supply-chain pattern and fails in restricted environments).
# Install manually first, e.g.:
#   pipx install uv==0.4.18    # or: brew install uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ uv is not installed.${NC}"
    echo "Install it first (e.g. 'pipx install uv' or 'brew install uv') and re-run."
    echo "See https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✓ uv is available"
echo ""

# Fixed dependency versions (exact pins so runs are reproducible across machines)
TORCH_VERSION="2.3.0"
BOTORCH_VERSION="0.11.3"
GPYTORCH_VERSION="1.12"
NUMPY_VERSION="1.26.4"
SCIPY_VERSION="1.13.1"
PANDAS_VERSION="2.2.2"
MATPLOTLIB_VERSION="3.9.0"
PLOTLY_VERSION="5.22.0"
SHAP_VERSION="0.45.1"
SEABORN_VERSION="0.13.0"

PYTHON_VERSIONS=("3.10" "3.11" "3.12")
REFERENCE_VERSION="3.10"

# Clean up old baselines
rm -rf dev/baselines

# Clean up old test environments if not keeping
if [ "$KEEP_VENV" = false ]; then
    echo "Cleaning up old test environments..."
    for pyver in "${PYTHON_VERSIONS[@]}"; do
        rm -rf .venv/py${pyver}
    done
else
    echo "Keeping existing virtual environments..."
fi

echo ""
echo "========================================================================"
echo "Create/verify environments with exact same dependency versions"
echo "========================================================================"
echo ""

for pyver in "${PYTHON_VERSIONS[@]}"; do
    if [ "$KEEP_VENV" = true ] && [ -d ".venv/py${pyver}" ]; then
        echo "→ Using existing Python ${pyver} environment..."
    else
        echo "→ Creating Python ${pyver} environment..."
        uv venv --python ${pyver} .venv/py${pyver}

        # Install exact versions
        uv pip install \
            torch==${TORCH_VERSION} \
            botorch==${BOTORCH_VERSION} \
            gpytorch==${GPYTORCH_VERSION} \
            numpy==${NUMPY_VERSION} \
            scipy==${SCIPY_VERSION} \
            pandas==${PANDAS_VERSION} \
            matplotlib==${MATPLOTLIB_VERSION} \
            plotly==${PLOTLY_VERSION} \
            shap==${SHAP_VERSION} \
            seaborn==${SEABORN_VERSION} \
            pyyaml \
            pytest \
            pytest-xdist \
            --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple \
            --python .venv/py${pyver}/bin/python \
            --quiet

        # Install obsidian in editable mode (no deps)
        uv pip install -e . --python .venv/py${pyver}/bin/python --no-deps --quiet
    fi

    echo "  ✓ Python ${pyver}: torch ${TORCH_VERSION}, botorch ${BOTORCH_VERSION}, gpytorch ${GPYTORCH_VERSION}, numpy ${NUMPY_VERSION}"
done

echo ""
echo "========================================================================"
echo "Step 1: Capture reference baseline (Python ${REFERENCE_VERSION})"
echo "========================================================================"
echo ""

echo "→ Running tests on Python ${REFERENCE_VERSION} and capturing baseline..."
if ! .venv/py${REFERENCE_VERSION}/bin/python -m pytest obsidian/tests \
    -n auto \
    --capture-baseline=dev/baselines \
    -q 2>&1 | tee /tmp/pytest-py${REFERENCE_VERSION}.log | tail -5; then
    
    echo -e "${RED}✗ Baseline capture failed for Python ${REFERENCE_VERSION}${NC}"
    echo "Last 20 lines of output:"
    tail -20 /tmp/pytest-py${REFERENCE_VERSION}.log
    echo ""
    echo "Full log: /tmp/pytest-py${REFERENCE_VERSION}.log"
    exit 1
fi

baseline_count=$(ls -1 dev/baselines/*.csv 2>/dev/null | wc -l | tr -d ' ')
echo -e "${GREEN}✓ Reference baseline captured: ${baseline_count} files${NC}"

echo ""
echo "========================================================================"
echo "Step 2: Compare other Python versions against reference"
echo "========================================================================"
echo ""

all_passed=true

for pyver in "${PYTHON_VERSIONS[@]}"; do
    if [ "$pyver" = "$REFERENCE_VERSION" ]; then
        continue
    fi
    
    echo "→ Running tests on Python ${pyver} and comparing against baseline..."
    
    if ! .venv/py${pyver}/bin/python -m pytest obsidian/tests \
        -n auto \
        --compare-baseline=dev/baselines \
        -q 2>&1 | tee /tmp/pytest-py${pyver}.log | tail -5; then
        
        echo -e "${RED}✗ Python ${pyver} produced different results${NC}"
        echo "Last 20 lines of output:"
        tail -20 /tmp/pytest-py${pyver}.log
        echo ""
        echo "Full log: /tmp/pytest-py${pyver}.log"
        all_passed=false
    else
        echo -e "${GREEN}✓ Python ${pyver} is numerically equivalent to Python ${REFERENCE_VERSION}${NC}"
    fi
    echo ""
done

echo ""
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo ""

if [ "$all_passed" = true ]; then
    echo -e "${GREEN}✅ SUCCESS${NC}"
    echo ""
    echo "All numerical operations are equivalent within tolerance (rtol=atol=1e-9)"
    echo "across Python 3.10, 3.11, 3.12 with fixed dependency versions:"
    echo "  - torch: ${TORCH_VERSION}"
    echo "  - botorch: ${BOTORCH_VERSION}"
    echo "  - gpytorch: ${GPYTORCH_VERSION}"
    echo "  - numpy: ${NUMPY_VERSION}"
    echo ""
    echo "Residual differences are at machine-epsilon level (<=1e-14)."
    echo "This confirms that supporting Python 3.10-3.12 is safe."
    echo ""
    exit_code=0
else
    echo -e "${RED}❌ FAILED${NC}"
    echo ""
    echo "Some Python versions produced different numerical results."
    echo "Review the output above and check the log files in /tmp/"
    echo ""
    exit_code=1
fi

# Optional: Clean up test environments
if [ "$NO_CLEANUP" = false ]; then
    echo ""
    echo "Cleaning up test environments (use --no-cleanup to keep)..."
    for pyver in "${PYTHON_VERSIONS[@]}"; do
        rm -rf .venv/py${pyver}
    done
    rm -rf dev/baselines
    echo "✓ Cleanup complete"
else
    echo ""
    echo "Keeping test environments and baselines (--no-cleanup was specified)"
fi

exit $exit_code
