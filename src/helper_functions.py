import os
import re
import re
from datetime import datetime

# Author: Katharine Leney, April 2025

# ======================================================
# Function to parse the model folder name to extract filter info
# ======================================================
def parse_model_folder(folder_name):
    """
    Args:
        folder_name (str): e.g. 'bertopic_cluster_model_from_2015_excl_2020_2021_20250426'

    Returns:
        str: Human-readable description
    """
    description_parts = []

    # Remove the date suffix (always 8 digits at end)
    folder_name = re.sub(r'_\d{8}$', '', folder_name)

    # Extract from year
    from_match = re.search(r'from_(\d{4})', folder_name)
    if from_match:
        from_year = from_match.group(1)
        description_parts.append(f"From {from_year}")

    # Extract up_to year
    upto_match = re.search(r'up_to_(\d{4})', folder_name)
    if upto_match:
        upto_year = upto_match.group(1)
        description_parts.append(f"Up to {upto_year}")

    # Extract excluded years
    excl_match = re.search(r'excl_([\d_]+)', folder_name)
    if excl_match:
        years = excl_match.group(1).split("_")
        description_parts.append(f"Excluding {', '.join(years)}")

    return " | ".join(description_parts)


# ======================================================
# Auto-generate clean topic labels
# ======================================================

def generate_clean_labels(topic_model, top_n_words=1):

    new_labels = {} 
    junk_topics = []
    topic_info = topic_model.get_topic_info()

    # Define known abbreviation corrections
    manual_replacements = {
        "Ndc": "New Distribution Capability",
        "Iosa": "IATA Operational Safety Audit",
        "Bcbp": "Bar-Coded Boarding Pass",
        "Ceiv": "Center of Excellence for Independent Validators",
        "Id": "ID"
    }

    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            new_labels[topic_id] = "NULL"
            continue
        
        words = topic_model.get_topic(topic_id)
        
        if not words or len(words) == 0:
            new_labels[topic_id] = ""
            continue
        
        keywords = [word.replace("_", " ").capitalize() for word, _ in words[:top_n_words]]

        # Junk detection
        if (all((w.strip() == "") for w in keywords)) or (all(wgt <= 1e-5 for _, wgt in words)):

            print("Topic ", topic_id, "identified as junk:")
            print(topic_model.get_topic(topic_id))
              
            new_labels[topic_id] = ""
            junk_topics.append(topic_id)
            continue

        # Otherwise, build a proper label
        label = ", ".join(keywords)

        # Apply manual corrections
        for short, full in manual_replacements.items():
            if short.lower() in label.lower():
                label = label.replace(short, full).replace(short.lower(), full)
        
        new_labels[topic_id] = label

    return new_labels, junk_topics