# ==================================================================
# RAG assistant to answer questions about opportunities in aviation
# Usage:
#    python src/assistant.py
#        -v | --verbose : verbose mode
#        -rebuild : forces rebuild of model
#
# Author: Katharine Leney, April 2025
# ==================================================================

from sentence_transformers import SentenceTransformer
import warnings
import random
import argparse
from utils.assistant_utils import load_summariser, summarise_texts, load_or_build_index, dynamic_top_k
from assistant_config import EMBEDDING_MODEL, TOP_K, TAKEOFF_MESSAGES, EXIT_MESSAGES

# Suppress annoying (but harmless!) transformer warnings
# Investigate this more later...
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pytorch_utils")


# --------------------------------------------------
# Function to retrieve most relevant chunks 
# of text related to user query
# --------------------------------------------------

def ask_question(model, index, texts, years, query, top_k=TOP_K, verbose=False):

    query_embedding = model.encode([query], convert_to_numpy=True)

    # D = distance (similarity) scores between query and top_n matches
    # I = Indices of the top_n closest matches to the query
    # Arrays, with shape (number of queries (1!), top_n)
    D, I = index.search(query_embedding, top_k)

    # Pair distance, text, and year
    results = [(dist, texts[idx], years[idx]) for idx, dist in zip(I[0], D[0])]
    
    # Sort by distance (lower = better match)
    results = sorted(results, key=lambda x: x[0])

    # Dynamically decide how many matches to keep
    # based on distance gap between elements
    dynamic_decision = True
    if dynamic_decision :
        selected_k = dynamic_top_k(D[0])

        # Select top selected_k results
        D = D[:, :selected_k]
        I = I[:, :selected_k]

    # If verbose mode, print the distance, year, and text snippet of top 5 matches.
    if verbose:
        print("\nTop Matches (Distance Scores):")
        for rank, (dist, text, year) in enumerate(results[:D[0].size], start=1):
            snippet = text[:50].replace('\n', ' ').strip() + ("..." if len(text) > 50 else "")
            print(f"{rank}. Distance: {dist:.4f} | Year: {year} | Text Snippet: \"{snippet}\"")

    # Return just (text, year) to the summariser
    cleaned_results = [(text, year) for _, text, year in results]

    return cleaned_results


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
    summariser = load_summariser()

    print("\n" + random.choice(TAKEOFF_MESSAGES))
    while True:
        query = input("Ask your question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            print("\n" + random.choice(EXIT_MESSAGES))
            break

        results = ask_question(model, index, texts, years, query, top_k=TOP_K, verbose=verbose_mode)
        retrieved_texts = [text for text, year in results]
        answer = summarise_texts(retrieved_texts, summariser)

        print("\n===== Assistant's Answer =====\n")
        print(answer)
        print("\n==============================\n")

if __name__ == "__main__":
    main()