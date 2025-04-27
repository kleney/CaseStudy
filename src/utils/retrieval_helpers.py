# ==================================================================
# Helper functions for retrieval mode of assistant.py
# Author: Katharine Leney, April 2025
# ==================================================================

import numpy as np
from transformers import pipeline
from assistant_config import SUMMARISER_MODEL, THRESHOLD, TOP_K

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