#!/bin/bash
# ICDI-X startup script
# Kills stale processes, clears Qdrant locks, then starts backend + frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🔴 Stopping any existing processes..."
pkill -9 -f uvicorn 2>/dev/null || true
pkill -9 -f "app.main" 2>/dev/null || true
pkill -9 -f "next dev" 2>/dev/null || true
pkill -9 -f "next-server" 2>/dev/null || true
# Also free ports explicitly
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :3000 | xargs kill -9 2>/dev/null || true
sleep 2

echo "🔑 Clearing Qdrant lock files..."
find "$SCRIPT_DIR/qdrant_data" -name "*.lock" -delete 2>/dev/null || true
find "$SCRIPT_DIR/qdrant_data" -name "lock" -delete 2>/dev/null || true
# Kill any process still holding the qdrant_data directory
lsof +D "$SCRIPT_DIR/qdrant_data" 2>/dev/null | awk 'NR>1{print $2}' | sort -u | xargs kill -9 2>/dev/null || true

echo "⚙️  Activating virtual environment..."
source "$SCRIPT_DIR/.venv/bin/activate"

echo "🚀 Starting backend on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

echo "⏳ Waiting for backend to be ready..."
for i in {1..15}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ Backend is up!"
        break
    fi
    sleep 1
done

echo "🌐 Starting frontend on port 3000..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "=========================================="
echo "✅ ICDI-X is running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   Backend logs: tail -f /tmp/backend.log"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait and handle Ctrl+C gracefully
trap "echo '🛑 Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
