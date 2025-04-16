#!/bin/bash
set -e

if [ "$1" = "text" ]; then
    echo "Running text extractor"
    exec python3.9 -m uvicorn src.models.text.app:app --host 0.0.0.0 --port 8000
elif [ "$1" = "image" ]; then
    echo "Running image extractor"
    exec python3.9 -m uvicorn src.models.image.app:app --host 0.0.0.0 --port 8000
elif [ "$1" = "both" ]; then
    echo "Running both text and image extractors"
    exec python3.9 -m uvicorn src.manager.app:app --host 0.0.0.0 --port 8000
else
    echo "Unknown command: $1"
    exit 1
fi
