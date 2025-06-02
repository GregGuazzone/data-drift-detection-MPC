# Get directory where script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Create log directory if it doesn't exist
mkdir -p ../../logs

# Define log file
LOG_FILE="../../logs/uk_$(date +%Y%m%d_%H%M%S).log"

echo "🇬🇧 Starting UK server (Party ID: 6)..."

# Run server based on arguments
if [ "$1" == "--send" ]; then
    python united_kingdom.py --send --target "$2" --message "$3" | tee "$LOG_FILE"
else
    python united_kingdom.py | tee "$LOG_FILE"
fi

echo "UK server stopped"