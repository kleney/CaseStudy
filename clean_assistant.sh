#!/usr/bin/env bash

# =========================================
# Deletes old RAG assistant models and 
# any HuggingFace cached models
# Author: Katharine Leney, April 2025
# =========================================

# check if the model info exists already, and if it does delete them
if [ -f "models/faiss_index.idx" ]; then
    echo "Deleting faiss_index.idx"
    rm models/faiss_index.idx
fi
if [ -f "models/texts_and_years.json" ]; then
    echo "Deleting texts_and_years.json"
    rm models/texts_and_years.json
fi
#huggingface-cli delete-cache --disable-tui
