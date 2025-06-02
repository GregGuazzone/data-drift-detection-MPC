# Get directory where script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Create log directory if it doesn't exist
mkdir -p ../../logs

# Define log file
LOG_FILE="../../logs/germany_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Germany server (Party ID: 2)..."

# Run server based on arguments
if [ "$1" == "--send" ]; then
    python germany.py --send --target "$2" --message "$3" | tee "$LOG_FILE"
else
    python germany.py | tee "$LOG_FILE"
fi

echo "Germany server stopped"