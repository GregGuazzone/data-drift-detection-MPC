# Get directory where script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Create log directory if it doesn't exist
mkdir -p ../../logs

# Define log file
LOG_FILE="../../logs/iran_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Iran server (Party ID: 3)..."

# Run server based on arguments
if [ "$1" == "--send" ]; then
    python iran.py --send --target "$2" --message "$3" | tee "$LOG_FILE"
else
    python iran.py | tee "$LOG_FILE"
fi

echo "Iran server stopped"