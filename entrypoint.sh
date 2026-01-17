#!/bin/bash
set -e

# Exemplo de uso: ./entrypoint.sh train
# Argumentos: train, evaluate, bash

case "$1" in
  train)
    shift
    python train.py "$@"
    ;;
  evaluate)
    shift
    python evaluate.py "$@"
    ;;
  bash)
    /bin/bash
    ;;
  *)
    echo "Uso: $0 {train|evaluate|bash} [args...]"
    exit 1
    ;;
esac
