# ==================================================================
# RAG (or retrieval-only) assistant to answer questions about 
# opportunities in aviation.
# Usage:
#    python src/assistant.py
#        -v | --verbose : verbose mode
#        -rebuild : forces rebuild of model
# Configure in src/assistant_config.py
# Helper functions in src/utils/assistant_utils.py and src/utils/rag_helper.py
#
# Author: Katharine Leney, April 2025
# ==================================================================

from sentence_transformers import SentenceTransformer
import warnings
import random
import argparse
import os
from assistant_config import EMBEDDING_MODEL, TOP_K, TAKEOFF_MESSAGES, EXIT_MESSAGES, RAG_MODE
from utils.assistant_helpers import load_or_build_index
from utils.retrieval_helpers import load_summariser, summarise_texts, ask_question
from utils.rag_helpers import handle_query_with_rag

# Suppress annoying (but harmless!) transformer warnings
# Investigate this more later...
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pytorch_utils")

# ---------------------------------
# Main assistant loop
# ---------------------------------
def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Aviation Assistant")
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild of the FAISS index and text store')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode (show top match distances)')
    args = parser.parse_args()

    force_rebuild = args.rebuild    
    verbose_mode = args.verbose

    print("\nLoading models and data...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    index, texts, years = load_or_build_index(model, force_rebuild=force_rebuild)
    #summariser = load_summariser()

    # Check if the OpenAI key is set if the user is in RAG mode
    # (doesn't help with checking paid credits... come back to this)
    global RAG_MODE  # allow us to modify RAG_MODE if needed
    if RAG_MODE :
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n[Error] RAG mode selected but no OpenAI API key found.")
            print("[Action] Switching to retrieval mode.\n")
            RAG_MODE = False

    print("\n" + random.choice(TAKEOFF_MESSAGES))
    while True:
        query = input("Ask your question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            print("\n" + random.choice(EXIT_MESSAGES))
            break

        # If RAG_MODE is enabled, use that, otherwise revert to
        # the retrieval-only method using the summariser
        if RAG_MODE :
            answer = handle_query_with_rag(query, model, index, texts)
        else :
            summariser = load_summariser()
            results = ask_question(model, index, texts, years, query, top_k=TOP_K, verbose=verbose_mode)
            retrieved_texts = [text for text, year in results]
            answer = summarise_texts(retrieved_texts, summariser)

        print("\n===== Assistant's Answer =====\n")
        print(answer)
        print("\n==============================\n")

if __name__ == "__main__":
    main()