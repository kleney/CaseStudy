# ==================================================================
# Helper functions for assistant.py
# Author: Katharine Leney, April 2025
# ==================================================================

import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from assistant_config import CHUNKS_FILE, INDEX_FILE, TEXTS_FILE, SUMMARISER_MODEL, THRESHOLD


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

# --------------------------------------------------
# Function to dynamically determine optimal number
# of documents to summarise, based on distance jumps
# threshold = minimum jump size to consider a 'gap'
# --------------------------------------------------
def dynamic_top_k(distances, threshold=THRESHOLD):

    distances = np.array(distances)
    diffs = np.diff(distances)

    for idx, diff in enumerate(diffs):
        #print(idx, ": Distance = ", distances[idx], " : Difference wrt next element: ", diffs[idx])
        if diff > threshold:
            #print ("Keeping ", idx+1, " text elements for summarising")
            return idx + 1 
        
    #print("No large gap found. Keeping all ", len(distances), "matches")
    return len(distances)


# --------------------------------------------------
# Load a lightweight summarisation model (T5-small) 
# for condensing retrieved texts
# --------------------------------------------------
def load_summariser():
    summariser = pipeline("summarization", model=SUMMARISER_MODEL, tokenizer=SUMMARISER_MODEL, framework="pt")
    return summariser


# --------------------------------------------------
# Summarise a list of texts into a concise answer 
# using the loaded summarisation model
# (Summariser reads left-to-right, first bits of
# text dominate results)
# --------------------------------------------------
def summarise_texts(texts, summariser):

    if not texts:
        return "No relevant information found."
    
    concatenated = " ".join(texts)

    if len(concatenated.split()) < 30:
        return concatenated.strip()
    
    if len(concatenated) > 2000:
        concatenated = concatenated[:2000]

    prompt = "summarize: " + concatenated
    summary = summariser(prompt, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    #summary = summariser(prompt)[0]['summary_text']
    return summary

# --------------------------------------------------
# Improved handling of text in summariser model.
# Tells the summariser to handle all provided text
# on an equal footing. 
# (Hoped that this would improve the performance of
# the summaries, but limitation seems to be the model)
# --------------------------------------------------
def summarise_texts_fairly(text_list, summariser, max_snippets=10, max_chars=2048):

    prompt = "Summarise the following snippets fairly:\n\n"

    for idx, text in enumerate(text_list[:max_snippets]):
        prompt += f"Snippet {idx+1}:\n{text}\n\n"

    # Truncate prompt if needed
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]

    prompt += "Summary:\n"

    summary = summariser(prompt)[0]['summary_text']

    return summary