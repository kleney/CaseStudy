# ==================================================================
# RAG assistant to answer questions about opportunities in aviation
# Run ./clean_assistant.sh before running this script if you want
# to retrain the model, otherwise it will load the existing one.
#
# PREREQUISITES: Run clustering_analysis_visualisation.ipynb to
# generate outputs/topics_over_time.csv and outputs/topic_labels.json
# These are needed for trend analysis.
#
# Author: Katharine Leney, April 2025
# ==================================================================

import os
import sys
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import pipeline

sys.path.append("../src")
from utils.build_trend_data import build_trend_data

# ---------------------------------
# Configuration
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
INDEX_FILE = os.path.join(MODEL_DIR, "faiss_index.idx")
TEXTS_FILE = os.path.join(MODEL_DIR, "texts_and_years.json")
TREND_DATA_FILE = os.path.join(DATA_DIR, "trend_data.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

TOPIC_ALIASES = {
    "sustainable aviation fuel": "SAF",
    "saf": "SAF",
    "new distribution capability": "New Distribution Capability",
    "ndc": "New Distribution Capability",
    "iata operational safety audit": "Safety",
    "iosa": "Safety",
    "bar-coded boarding pass": "Bar-Coded Boarding Pass",
    "bcbp": "Bar-Coded Boarding Pass",
    "center of excellence for independent validators": "Center of Excellence for Independent Validators",
    "ceiv": "Center of Excellence for Independent Validators",
    "identity": "ID",
    "digital identity": "ID",
    "slot management": "Slot",
    "passenger rights": "Regulation",
    "regulation": "Regulation",
    "taxation": "Tax",
    "diversity and inclusion": "Diversity",
    "lithium batteries": "Lithium",
}

# ---------------------------------
# Helper Functions
# ---------------------------------

def load_chunks(filepath):
    with open(filepath, "r") as f:
        chunks = json.load(f)
    texts = [chunk["text"] for chunk in chunks]
    years = [chunk.get("year", 0) for chunk in chunks]
    return texts, years


def load_or_build_index(model):
    if os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE):
        print("Loading saved FAISS index and texts...")
        index = faiss.read_index(INDEX_FILE)
        with open(TEXTS_FILE, "r") as f:
            saved = json.load(f)
        texts = [item["text"] for item in saved]
        years = [item["year"] for item in saved]
    else:
        print("Building new FAISS index...")
        texts, years = load_chunks(CHUNKS_FILE)
        embeddings = []
        for text in tqdm(texts, desc="Embedding texts"):
            embeddings.append(model.encode(text, convert_to_numpy=True))
        embeddings = np.vstack(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        faiss.write_index(index, INDEX_FILE)
        with open(TEXTS_FILE, "w") as f:
            json.dump([{"text": t, "year": y} for t, y in zip(texts, years)], f)

    return index, texts, years


def ask_question(model, index, texts, years, query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)
    results = [(texts[i], years[i]) for i in I[0]]
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results


def load_summariser():
    summariser = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt")
    #summariser = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn", framework="pt")
    return summariser


def summarise_texts(texts, summariser):
    concatenated = " ".join(texts)
    if len(concatenated.split()) < 30:
        # If too short, just return raw text without summarising
        return concatenated.strip()
    if len(concatenated) > 2000:
        concatenated = concatenated[:2000]  # Truncate if too long
    prompt = "summarize: " + concatenated
    summary = summariser(prompt, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary


def load_trend_data(filepath):
    if not os.path.exists(filepath):
        print("\nTrend data not found, attempting to generate...")
        try:
            build_trend_data(
                topics_over_time_file=os.path.join(OUTPUTS_DIR, "topics_over_time.csv"),
                topic_labels_file=os.path.join(OUTPUTS_DIR, "topic_labels.json"),
                trend_data_output_file=filepath
            )
        except Exception as e:
            print(f"Failed to build trend data: {e}")
            return None

    try:
        with open(filepath, "r") as f:
            trend_data = json.load(f)
        return trend_data
    except Exception as e:
        print(f"Error loading trend data: {e}")
        return None

TREND_KEYWORDS = ["growth", "trend", "increase", "decrease", "over time", "change", "evolution"]

def is_trend_question(query):
    query = query.lower()
    return any(keyword in query for keyword in TREND_KEYWORDS)


def answer_trend_question(query, trend_data):
    query_lower = query.lower()
    matching_topics = []

    # First check aliases
    for alias, standard_topic in TOPIC_ALIASES.items():
        if alias in query_lower:
            if standard_topic in trend_data:
                matching_topics.append(standard_topic)

    # If no alias match, fallback to direct label search
    if not matching_topics:
        for topic_name in trend_data.keys():
            if topic_name.lower() in query_lower:
                matching_topics.append(topic_name)

    if not matching_topics:
        return "Sorry, I couldn't find a clear trend match. Please try rephrasing."

    answers = []
    for topic in matching_topics:
        data = trend_data[topic]
        years = sorted(data.keys())
        if len(years) >= 2:
            start, end = years[0], years[-1]
            start_val, end_val = data[start], data[end]
            growth_rate = ((end_val - start_val) / (start_val + 1e-5)) * 100
            answers.append(f"{topic}: {start_val:.3f} in {start} → {end_val:.3f} in {end} ({growth_rate:.1f}% growth)")
        else:
            answers.append(f"{topic}: insufficient data for growth calculation.")

    return "\n".join(answers)


# ---------------------------------
# Main Assistant Loop
# ---------------------------------

def main():
    print("\nLoading models and data...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    index, texts, years = load_or_build_index(model)
    summariser = load_summariser()
    trend_data = load_trend_data(TREND_DATA_FILE)

    if trend_data is None:
        print("\nWarning: Trend data unavailable. Only semantic retrieval will be available.")

    print("\nAssistant ready for takeoff!\n")
    while True:
        query = input("Ask your question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            print("\nWe hope you enjoyed flying with us today. Goodbye!")
            break

        if trend_data and is_trend_question(query):
            answer = answer_trend_question(query, trend_data)
        else:
            results = ask_question(model, index, texts, years, query, top_k=TOP_K)
            retrieved_texts = [text for text, year in results]
            answer = summarise_texts(retrieved_texts, summariser)

        print("\n===== Assistant's Answer =====\n")
        print(answer)
        print("\n==============================\n")

if __name__ == "__main__":
    main()