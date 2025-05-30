
# ================================================================
# Processes all pdf files in a given folder, applies some basic
# cleaning, splits them into chunks of max 1000 characters
# following the paragraph structure of the original text and 
# writes out a json file (data/chunks.json) for further 
# processing and analysis.
# Author: Katharine Leney, April 2025
# ================================================================

import os
import fitz  # from PyMuPDF, to open and extract text from pdf files
import json
import re    # regular expressions to extract years from filenames
import sys
import unicodedata

_PUNCT_CHARS = {chr(i) for i in range(sys.maxunicode)
                if unicodedata.category(chr(i)).startswith("P")}

# ----------------------------------------------------------------
# Remove single letter words but keep hyphenated letters
# e.g. "x" would be removed, but "x-ray" would be kept
# ----------------------------------------------------------------
def remove_single_letter_tokens(text: str) -> str:
    """
    Splits on whitespace, strips any leading/trailing punctuation
    (including Unicode punctuation) from each token, and drops it
    if what's left is exactly one ASCII letter.
    """
    out = []
    for tok in text.split():
        # strip off *all* leading/trailing punctuation
        # (we do this manually rather than str.strip() so we catch Unicode too)
        start, end = 0, len(tok)
        while start < end and tok[start] in _PUNCT_CHARS:
            start += 1
        while end > start and tok[end-1] in _PUNCT_CHARS:
            end -= 1
        core = tok[start:end]

        # if the core is exactly one ASCII letter, drop it
        if len(core) == 1 and core.isalpha():
            continue

        out.append(tok)
    return " ".join(out)

# ----------------------------------------------------------------
# Handle possessives properly
# ----------------------------------------------------------------
def clean_pdf_text(text: str) -> str:
    # Normalise apostrophes
    text = text.replace("’", "'")
    # Strip simple possessives: word’s --> word
    text = re.sub(r"\b(\w+)'s\b", r"\1", text, flags=re.IGNORECASE)
    return text

# ----------------------------------------------------------------
# Extract and process text from all PDFs in the given directory
# ----------------------------------------------------------------
def extract_text_from_pdfs(pdf_dir):
    chunks = []

    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue

        year = extract_year(fname)
        path = os.path.join(pdf_dir, fname)
        doc = fitz.open(path)
        full_text = ""

        for page in doc:
            # Clean and flatten text from each page
            text = page.get_text()
            text = clean_pdf_text(text)
            text = text.replace("-\n", "")  # Fix hyphenated line breaks
            text = text.replace("\n", " ")  # Replace hard newlines with space
            full_text += text + "\n\n"      # Simulate paragraph spacing between pages

        # Chunk the cleaned text
        split_chunks = split_into_chunks(full_text, max_length=1000)
        for chunk in split_chunks:
            chunks.append({"year": year, "text": chunk})

    return chunks

# ----------------------------------------------------------------
# Extracts the first 4-digit number from the filename as the year
# ----------------------------------------------------------------
def extract_year(filename):
    match = ''.join(c for c in filename if c.isdigit())
    if len(match) >= 4:
        return match[:4]
    return "unknown"

# ----------------------------------------------------------------
# Splits text into chunks of approximately max_length characters,
# preferring paragraph and sentence boundaries
# ----------------------------------------------------------------
def split_into_chunks(text, max_length=1000):

    # Split text into paragraphs (based on double newlines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:

        # Remove any leading or trailing characters from the paragraph string
        para = para.strip(" ,.-–—‘“”\"")

        # Remove single letter words
        para = remove_single_letter_tokens(para)

        # Accumulate paragraphs until close to max_length
        if len(current_chunk) + len(para) + 2 < max_length:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            if len(para) >= max_length:
                # Split long paragraphs on sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', para)
                buffer = ""
                for sentence in sentences:
                    if len(buffer) + len(sentence) + 1 < max_length:
                        buffer += sentence + " "
                    else:
                        chunks.append(buffer.strip())
                        buffer = sentence + " "
                if buffer.strip():
                    chunks.append(buffer.strip())
                current_chunk = ""
            else:
                current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(ROOT, "data", "annual_reviews")
    output_path = os.path.join(ROOT, "data", "chunks.json")

    # Extract and chunk the text
    chunks = extract_text_from_pdfs(input_dir)

    # Write to JSON file
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
