import os

# ==================================================================
# Configuration settings for assistant.py
# Author: Katharine Leney, April 2025
# ==================================================================

# ---------------------------------
# Base directories
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "assistant")

# ---------------------------------
# Ensure MODEL_DIR exists
# ---------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------
# Key files
# ---------------------------------
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
INDEX_FILE = os.path.join(MODEL_DIR, "faiss_index.idx")
TEXTS_FILE = os.path.join(MODEL_DIR, "texts_and_years.json")

# ---------------------------------
# Run in RAG or Retrieval-only mode
# ---------------------------------
RAG_MODE = True

# ---------------------------------
# Model names
# ---------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # rebuild if this is changed
SUMMARISER_MODEL = "t5-base" # t5-small also an option (no need to rebuild switching these)
RAG_MODEL = "gpt-3.5-turbo" # Requires setting up an OpenAI API key in your system and an OpenAI account

# ---------------------------------
# Retrieval settings
# ---------------------------------
TOP_K = 10 # Number of "top matches" to pass to function that evaluates how many text snippets should be summarised
THRESHOLD = 0.1 # "Gap" setting for determining cut-off point for clustering of text snippets

# ---------------------------------
# Aviation-themed messages
# for the assistant to use
# ---------------------------------
TAKEOFF_MESSAGES = [
    "Assistant ready for takeoff!",
    "All systems go. Ready for boarding!",
    "Cleared for departure. Standing by for your command!",
    "Flight plan uploaded. Awaiting your destination!",
]

EXIT_MESSAGES = [
    "We hope you enjoyed flying with us today. Goodbye!",
    "Landing successful. Thanks for flying with us!",
    "Your journey is complete. Farewell!",
    "Cabin crew disarm doors and cross-check. Bye!",
]
