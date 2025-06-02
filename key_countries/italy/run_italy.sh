# Get directory where script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Create log directory if it doesn't exist
mkdir -p ../../logs

# Define log file
LOG_FILE="../../logs/italy_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Italy server (Party ID: 4)..."

# Run server based on arguments
if [ "$1" == "--send" ]; then
    python italy.py --send --target "$2" --message "$3" | tee "$LOG_FILE"
else
    python italy.py | tee "$LOG_FILE"
fi

echo "Italy server stopped"