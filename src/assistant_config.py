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
# Model names
# ---------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # rebuild if this is changed
SUMMARISER_MODEL = "t5-base" # t5-small also an option (no need to rebuild switching these)

# ---------------------------------
# Retrieval settings
# ---------------------------------
TOP_K = 100

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
