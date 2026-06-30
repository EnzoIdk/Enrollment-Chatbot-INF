#!/bin/bash

echo "Iniciando Ollama en segundo plano:"
nohup ollama serve > ollama.log 2>&1 &

echo "Iniciando el API para el bot de Discord:"
source ./.venv/bin/activate
python ./main.py
