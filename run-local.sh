#!/usr/bin/env bash
set -e

# 1) lance Ollama si pas déjà lancé
if ! nc -z localhost 11434 2>/dev/null; then
  echo "→ start ollama serve"
  (ollama serve >/tmp/ollama.log 2>&1 &) 
  sleep 1
fi

# 2) exporte OLLAMA_HOST pour cette session
export OLLAMA_HOST="http://localhost:11434"

# 3) lance le proxy
echo "→ start proxy (http://localhost:3000)"
node server.js
