#!/bin/bash

conda activate base
conda remove -n od-rag --all --yes
conda env create -f src/setup/requirements.yaml
conda activate od-rag
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3