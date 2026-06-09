#!/bin/bash
set -e

IOPAINT_DEVICE="${IOPAINT_DEVICE:-cuda}"

echo "Iniciando IOPaint (LaMa) en puerto 8080 con device=${IOPAINT_DEVICE} ..."
iopaint start --model=lama --device="${IOPAINT_DEVICE}" --host=0.0.0.0 --port=8080 &
IOPAINT_PID=$!

trap 'kill -TERM "${IOPAINT_PID}" 2>/dev/null || true' EXIT TERM INT

echo "Iniciando API FastAPI en puerto 8000 ..."
exec uvicorn service:app --host 0.0.0.0 --port 8000
