#!/bin/bash

# Dualist Active Learning System - Quick Start Script

echo "======================================"
echo "Dualist Active Learning + LLM"
echo "Quick Start Script"
echo "======================================"
echo ""

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY is not set!"
    echo "    Set it with: export OPENAI_API_KEY='your-key-here'"
    echo ""
fi

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "⚠️  Port $1 is already in use. Please stop the service or use a different port."
        return 1
    fi
    return 0
}

# Check ports
check_port 8000 || exit 1
check_port 8080 || exit 1

echo "Starting system components..."
echo ""

# Start FastAPI backend
echo "📡 Starting FastAPI backend on port 8000..."
cd python-backend
python3 -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!
echo "   FastAPI PID: $FASTAPI_PID"
cd ..

# Wait for backend to start
sleep 3

# Start Django frontend
echo "🌐 Starting Django frontend on port 8080..."
cd django-frontend
python3 manage.py migrate > /dev/null 2>&1
python3 manage.py runserver 0.0.0.0:8080 &
DJANGO_PID=$!
echo "   Django PID: $DJANGO_PID"
cd ..

echo ""
echo "======================================"
echo "✅ System is starting up!"
echo "======================================"
echo ""
echo "Services:"
echo "  - FastAPI Backend:  http://localhost:8000"
echo "  - API Docs:         http://localhost:8000/docs"
echo "  - Django Frontend:  http://localhost:8080"
echo ""
echo "To stop the system:"
echo "  kill $FASTAPI_PID $DJANGO_PID"
echo ""
echo "Or press Ctrl+C to stop all services"
echo ""
echo "Waiting for services to start..."
sleep 5

# Test backend connection
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ Backend is running!"
else
    echo "❌ Backend failed to start. Check logs above."
fi

# Test frontend connection
if curl -s http://localhost:8080/ > /dev/null; then
    echo "✅ Frontend is running!"
else
    echo "❌ Frontend failed to start. Check logs above."
fi

echo ""
echo "======================================"
echo "Ready! Open http://localhost:8080"
echo "======================================"

# Wait for user interrupt
wait
