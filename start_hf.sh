#!/usr/bin/env bash
set -euo pipefail

PORT_NUM="${PORT:-7860}"

# Start internal REST API v4.0 microservice on port 8502 in background
python /app/api_server.py &

# Launch Streamlit UI on Hugging Face Spaces port 7860
exec python -m streamlit run /app/app.py \
    --server.port="${PORT_NUM}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
