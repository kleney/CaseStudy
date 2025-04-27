
# ==================================================================
# General helper functions for assistant.py
# Author: Katharine Leney, April 2025
# ==================================================================

import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from assistant_config import CHUNKS_FILE, INDEX_FILE, TEXTS_FILE

# --------------------------------------------------
# Load data/chunks.json file
# (produced by src/extract.py)
# --------------------------------------------------
def load_chunks(filepath):
    with open(filepath, "r") as f:
        chunks = json.load(f)
    texts = [chunk["text"] for chunk in chunks]
    years = [chunk.get("year", 0) for chunk in chunks]
    return texts, years

# --------------------------------------------------
# Load/build the model
# --------------------------------------------------
def load_or_build_index(model, force_rebuild=False):

    index_exists = os.path.exists(INDEX_FILE)
    texts_exists = os.path.exists(TEXTS_FILE)

    # Build the model if the force_rebuild argument is set, 
    # or if model files are not found.  
    if force_rebuild or not (index_exists and texts_exists):
        if force_rebuild:
            print("Rebuild requested: deleting old model files if they exist...")
        else:
            print("Model files missing: building new FAISS index...")

        # Delete old files if they exist
        try:
            if os.path.exists(INDEX_FILE):
                os.remove(INDEX_FILE)
            if os.path.exists(TEXTS_FILE):
                os.remove(TEXTS_FILE)
        except Exception as e:
            print(f"Warning: problem deleting old files: {e}")

        # Build from scratch
        texts, years = load_chunks(CHUNKS_FILE)
        embeddings = []
        for text in tqdm(texts, desc="Embedding texts"):
            embeddings.append(model.encode(text, convert_to_numpy=True))
        embeddings = np.vstack(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Write faiss_index.idx and texts_and_years.json to
        # models/assistant directory.
        faiss.write_index(index, INDEX_FILE)
        with open(TEXTS_FILE, "w") as f:
            json.dump([{"text": t, "year": y} for t, y in zip(texts, years)], f)

    # Otherwise, load the existing model.
    else:
        print("Loading existing FAISS index and texts...")
        index = faiss.read_index(INDEX_FILE)
        with open(TEXTS_FILE, "r") as f:
            saved = json.load(f)
        texts = [item["text"] for item in saved]
        years = [item["year"] for item in saved]

    return index, texts, years