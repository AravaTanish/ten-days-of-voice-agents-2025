#!/bin/bash

# Start all services in background
livekit-server --dev &
(cd backend && uv run python src/fraudAlertAgent.py dev) &
(cd frontend && pnpm dev) &

# Wait for all background jobs
wait