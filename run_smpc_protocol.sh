#!/bin/bash
echo "Starting SMPC Protocol for 2021-01-01 Daily Cases"
echo "======================================================"

# Configuration
COUNTRIES=("china" "france" "germany" "iran" "italy" "spain" "united_kingdom" "us")
TARGET_DATE="2021-01-01"
DATA_COLUMN="daily_cases"
RESULTS_DIR="./results"

# Create logs directory
mkdir -p logs
mkdir -p $RESULTS_DIR

# Store PIDs
declare -a SERVER_PIDS

# Function to clean up on exit
cleanup() {
    echo ""
    echo "Shutting down all servers..."
    for pid in "${SERVER_PIDS[@]}"; do
        [ ! -z "$pid" ] && kill $pid 2>/dev/null
    done
    echo "All servers stopped."
    exit 0
}

# Set trap for clean exit
trap cleanup INT

# Kill any existing python processes that might be using our ports
echo "Cleaning up existing processes..."
pkill -f "python.*\.py" || true
sleep 2

# Start all countries with COMBINED server+protocol mode
echo "Starting all countries in combined server+protocol mode..."

# Start China (dealer) first to ensure it's ready
echo "Starting China (dealer)..."
cd "key_countries/china"
python3 "china.py" --run-smpc --date $TARGET_DATE --column $DATA_COLUMN > "../../logs/china_combined.log" 2>&1 &
CHINA_PID=$!
SERVER_PIDS+=($CHINA_PID)
cd - > /dev/null
echo "Started China (PID: $CHINA_PID)"
sleep 5  # Give dealer time to initialize

# Start all other countries with combined mode
for country in "${COUNTRIES[@]}"; do
    if [ "$country" != "china" ]; then
        echo "Starting ${country}..."
        
        cd "key_countries/${country}"
        python3 "${country}.py" --run-smpc --date $TARGET_DATE --column $DATA_COLUMN > "../../logs/${country}_combined.log" 2>&1 &
        pid=$!
        SERVER_PIDS+=($pid)
        cd - > /dev/null
        
        echo "Started ${country} (PID: $pid)"
        sleep 2  # Small delay between country starts
    fi
done

# Wait for protocol to complete
echo "Protocol initiated for all countries"
echo "Waiting for protocol to complete (this may take a minute)..."

# Give time for protocol to complete
sleep 100

# Check for result file
RESULT_FILE="key_countries/china/results/smpc_result_${TARGET_DATE//-/}.csv"
if [ -f "$RESULT_FILE" ]; then
    echo "Protocol completed successfully!"
    echo "Results:"
    cat "$RESULT_FILE"
    cp "$RESULT_FILE" "$RESULTS_DIR/"
    echo "Results saved to: $RESULTS_DIR/$(basename $RESULT_FILE)"
else
    echo "No result file found. Protocol may not have completed."
    echo "Check logs for errors:"
    echo "  tail logs/china_combined.log"
    
    # Look for common errors in various logs
    echo "Checking all logs for errors:"
    grep -i "error\|exception\|failed" logs/*combined.log | tail -10
fi

echo ""
echo "Press Ctrl+C to stop all servers and exit"

# Keep script running until user presses Ctrl+C
while true; do
    sleep 1
done