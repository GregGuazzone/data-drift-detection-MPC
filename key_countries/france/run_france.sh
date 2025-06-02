# Get directory where script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Create log directory if it doesn't exist
mkdir -p ../../logs

# Define log file
LOG_FILE="../../logs/france_$(date +%Y%m%d_%H%M%S).log"

echo "Starting France server (Party ID: 1)..."

# Run server based on arguments
if [ "$1" == "--send" ]; then
    python france.py --send --target "$2" --message "$3" | tee "$LOG_FILE"
else
    python france.py | tee "$LOG_FILE"
fi

echo "France server stopped"