# src/utils/generate_clean_labels.py
# ======================================================
# Helper function to auto-generate clean topic labels
# Author: Katharine Leney, April 2025 (extended)
# ======================================================

from utils.topic_map import custom_map

def generate_clean_labels(topic_model, top_n_words=1):

    new_labels = {}
    junk_topics = []
    topic_info = topic_model.get_topic_info()

    # Known abbreviation corrections
    manual_replacements = {
        "ndc": "New Distribution Capability",
        "iosa": "IATA Operational Safety Audit",
        "bcbp": "Bar-Coded Boarding Pass",
        "ceiv": "Center of Excellence for Independent Validators",
        "id": "ID",
        "saf": "SAF"
    }

    # Topic -1 is always junk by definition
    for topic_id in topic_info["Topic"]:
        # Mark -1 as null label
        if topic_id == -1:
            new_labels[topic_id] = "NULL"
            continue

        # Get the raw topic name (with index prefix)
        raw_name = topic_info.loc[topic_info["Topic"] == topic_id, "Name"].iloc[0]
        # Remove numeric prefix (everything up to first underscore)
        if "_" in raw_name:
            suffix = raw_name.split("_", 1)[1]
        else:
            suffix = raw_name

        # Check for any custom_map substring match
        for key, custom_label in custom_map.items():
            if key in suffix:
                new_labels[topic_id] = custom_label
                break
        else:
            # No custom match: fall back to token-based labelling
            words = topic_model.get_topic(topic_id)
            if not words:
                new_labels[topic_id] = ""
                continue

            # Build label from top_n_words tokens
            keywords = []
            for word, _ in words[:top_n_words]:
                if word in manual_replacements:
                    token = manual_replacements[word]
                else:
                    token = word.replace("_", " ").capitalize()
                keywords.append(token)

            # Junk detection 
            weights = [w for _, w in words[:top_n_words]]
            if all((k.strip() == "") for k in keywords) or all(w <= 1e-5 for w in weights):
                print(f"Topic {topic_id} identified as junk:")
                print(topic_model.get_topic(topic_id))
                new_labels[topic_id] = ""
                junk_topics.append(topic_id)
                continue

            # Build the labels
            new_labels[topic_id] = ", ".join(keywords)

    return new_labels, junk_topics
