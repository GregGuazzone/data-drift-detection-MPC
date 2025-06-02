# Get directory where script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Create log directory if it doesn't exist
mkdir -p ../../logs

# Define log file
LOG_FILE="../../logs/us_$(date +%Y%m%d_%H%M%S).log"

echo "🇺🇸 Starting US server (Party ID: 7)..."

# Run server based on arguments
if [ "$1" == "--send" ]; then
    python us.py --send --target "$2" --message "$3" | tee "$LOG_FILE"
else
    python us.py | tee "$LOG_FILE"
fi

echo "US server stopped"