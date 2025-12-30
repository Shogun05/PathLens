#!/bin/bash
# PathLens Backend Server Startup Script

cd "$(dirname "$0")"

echo "Starting PathLens Backend API..."
echo "========================================"
echo "API will be available at: http://localhost:8001"
echo "API Documentation: http://localhost:8001/docs"
echo "========================================"
echo ""

# Check if dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    python3 -m pip install -q -r requirements.txt
    echo "Dependencies installed."
fi

# Start the server with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
