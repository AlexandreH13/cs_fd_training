#!/bin/bash

SRC_DIR="/app/"

# Alterar para o diretório 'src'
cd "$SRC_DIR" || { echo "Erro: não foi possível acessar o diretório $SRC_DIR"; exit 1; }

echo "Iniciando o treinamento..."
python3 src/train/train.py