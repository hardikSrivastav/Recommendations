#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
COVERAGE=true
PARALLEL=false
TEST_CATEGORY="all"
VERBOSE=false
FAILFAST=false

# Help message
show_help() {
    echo "Usage: ./run_tests.sh [OPTIONS]"
    echo "Run tests for the music recommendation system"
    echo
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -c, --no-coverage          Run tests without coverage report"
    echo "  -p, --parallel             Run tests in parallel"
    echo "  -v, --verbose              Show verbose output"
    echo "  -f, --fail-fast            Stop on first test failure"
    echo "  -t, --type TYPE            Run specific test category:"
    echo "                             - all (default)"
    echo "                             - auth (Authentication tests)"
    echo "                             - database (Database service tests)"
    echo "                             - feedback (Feedback route tests)"
    echo "                             - models (ML model tests)"
    echo "                             - predictors (Predictor tests)"
    echo "                             - learning (Continuous learning tests)"
    echo
    echo "Examples:"
    echo "  ./run_tests.sh                     # Run all tests with coverage"
    echo "  ./run_tests.sh -p -v               # Run all tests in parallel with verbose output"
    echo "  ./run_tests.sh -t database         # Run only database tests"
    echo "  ./run_tests.sh -t feedback -f      # Run feedback tests and stop on first failure"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--no-coverage)
            COVERAGE=false
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--fail-fast)
            FAILFAST=true
            shift
            ;;
        -t|--type)
            TEST_CATEGORY="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate test category
valid_categories=("all" "auth" "database" "feedback" "models" "predictors" "learning")
if [[ ! " ${valid_categories[@]} " =~ " ${TEST_CATEGORY} " ]]; then
    echo -e "${RED}Error: Invalid test category '${TEST_CATEGORY}'${NC}"
    show_help
    exit 1
fi

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python -m venv venv
fi

# Install test requirements
echo -e "${BLUE}Installing test requirements...${NC}"
pip install -r requirements-test.txt

# Add parent directory to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(dirname $(pwd))"
echo -e "${BLUE}PYTHONPATH set to: $PYTHONPATH${NC}"

# Install the application package in development mode
echo -e "${BLUE}Installing application package...${NC}"
pip install -e ..

# Build the pytest command
PYTEST_CMD="pytest"

# Add category-specific test files
case $TEST_CATEGORY in
    "auth")
        PYTEST_CMD="$PYTEST_CMD test_auth.py"
        ;;
    "database")
        PYTEST_CMD="$PYTEST_CMD test_database_service.py"
        ;;
    "feedback")
        PYTEST_CMD="$PYTEST_CMD test_feedback_routes.py"
        ;;
    "models")
        PYTEST_CMD="$PYTEST_CMD test_model.py"
        ;;
    "predictors")
        PYTEST_CMD="$PYTEST_CMD test_predictors.py"
        ;;
    "learning")
        PYTEST_CMD="$PYTEST_CMD test_continuous_learning.py"
        ;;
    "all")
        PYTEST_CMD="$PYTEST_CMD"
        ;;
esac

# Add options based on flags
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=application --cov-report=html:coverage --cov-report=term-missing"
fi

if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v -s"
fi

if [ "$FAILFAST" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -x"
fi

# Add common options
PYTEST_CMD="$PYTEST_CMD --disable-warnings"

# Print test configuration
echo -e "${YELLOW}Test Configuration:${NC}"
echo -e "  Category:  ${BLUE}$TEST_CATEGORY${NC}"
echo -e "  Coverage:  ${BLUE}$([ "$COVERAGE" = true ] && echo "enabled" || echo "disabled")${NC}"
echo -e "  Parallel:  ${BLUE}$([ "$PARALLEL" = true ] && echo "enabled" || echo "disabled")${NC}"
echo -e "  Verbose:   ${BLUE}$([ "$VERBOSE" = true ] && echo "enabled" || echo "disabled")${NC}"
echo -e "  FailFast:  ${BLUE}$([ "$FAILFAST" = true ] && echo "enabled" || echo "disabled")${NC}"
echo
echo -e "${YELLOW}Running tests with command:${NC}"
echo -e "${BLUE}$PYTEST_CMD${NC}"
echo

# Run the tests
$PYTEST_CMD

# Store the exit status
TEST_EXIT_STATUS=$?

# Print summary
echo
if [ $TEST_EXIT_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed successfully!${NC}"
else
    echo -e "${RED}✗ Some tests failed!${NC}"
fi

# Open coverage report if generated and on macOS
if [ "$COVERAGE" = true ] && [ "$(uname)" == "Darwin" ] && [ $TEST_EXIT_STATUS -eq 0 ]; then
    echo -e "${BLUE}Opening coverage report...${NC}"
    open coverage/index.html
fi

# Deactivate virtual environment
deactivate

# Print final message
echo
if [ $TEST_EXIT_STATUS -eq 0 ]; then
    echo -e "${GREEN}Test suite completed successfully!${NC}"
else
    echo -e "${RED}Test suite failed with status $TEST_EXIT_STATUS${NC}"
fi

exit $TEST_EXIT_STATUS 