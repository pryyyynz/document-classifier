#!/bin/bash

# Contract Classification API Startup Script

echo "🚀 Starting Contract Classification API..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Install dependencies
echo "🔍 Checking dependencies..."
if [ -f "requirements.txt" ]; then
    echo "📦 Installing from requirements.txt..."
    python3 -m pip install -r requirements.txt
else
    echo "📦 requirements.txt not found. Installing minimal runtime deps..."
    python3 -m pip install fastapi uvicorn[standard] python-multipart python-docx pdfplumber pytesseract pillow opencv-python-headless numpy pandas scikit-learn lime python-docx pyyaml
fi

# Check if model exists
if [ ! -f "enhanced_models_output/models/enhanced_tfidf_gradient_boosting_model.pkl" ]; then
    echo "❌ Model not found. Please train the model first."
    echo "Run: python3 src/enhanced_training_pipeline.py"
    exit 1
fi

# Check if explainability module exists
if [ ! -f "src/explainability.py" ]; then
    echo "❌ Explainability module not found."
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "✅ Dependencies and model verified"

# Start the API server in background and wait for readiness
LOG_FILE="logs/api.log"
touch "$LOG_FILE"
(
  cd api
  python3 main.py
) >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Poll health endpoint until up (timeout ~60s)
echo "⏳ Waiting for server to become ready..."
ATTEMPTS=0
until curl -sSf http://localhost:8000/health >/dev/null 2>&1; do
  sleep 1
  ATTEMPTS=$((ATTEMPTS+1))
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server process terminated unexpectedly. Showing last logs:"
    tail -n 200 "$LOG_FILE" || true
    exit 1
  fi
  if [ $ATTEMPTS -ge 60 ]; then
    echo "❌ Server did not become ready within 60 seconds. Showing logs:"
    tail -n 200 "$LOG_FILE" || true
    kill $SERVER_PID 2>/dev/null || true
    exit 1
  fi
done

# On ready, print server info then stream logs
echo "🌐 Starting API server on http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Begin streaming logs and keep attached
tail -f "$LOG_FILE" &
TAIL_PID=$!

# Ensure cleanup on exit signals
cleanup() {
  kill $TAIL_PID 2>/dev/null || true
  kill $SERVER_PID 2>/dev/null || true
}
trap cleanup INT TERM

# Wait for the server process to exit, then cleanup tail
wait $SERVER_PID
cleanup
