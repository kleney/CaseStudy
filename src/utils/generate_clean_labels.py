
# ======================================================
# Helper function to auto-generate clean topic labels
# Author: Katharine Leney, April 2025
# ======================================================

def generate_clean_labels(topic_model, top_n_words=1):

    new_labels = {} 
    junk_topics = []
    topic_info = topic_model.get_topic_info()

    # Define known abbreviation corrections
    manual_replacements = {
        "ndc": "New Distribution Capability",
        "iosa": "IATA Operational Safety Audit",
        "bcbp": "Bar-Coded Boarding Pass",
        "ceiv": "Center of Excellence for Independent Validators",
        "id": "ID",
        "saf": "SAF"
    }

    for topic_id in topic_info["Topic"]:

        # Topic -1 is always junk by definition
        if topic_id == -1:
            new_labels[topic_id] = "NULL"
            continue
        
        words = topic_model.get_topic(topic_id)
        
        if not words or len(words) == 0:
            new_labels[topic_id] = ""
            continue
        
        keywords = []
        for word, _ in words[:top_n_words]:
            # Only replace full word matches
            if word in manual_replacements:
                word = manual_replacements[word]
            else:
                word = word.replace("_", " ").capitalize()
            keywords.append(word)

        # Junk detection
        if (all((w.strip() == "") for w in keywords)) or (all(wgt <= 1e-5 for _, wgt in words)):
            print("Topic ", topic_id, "identified as junk:")
            print(topic_model.get_topic(topic_id))
            new_labels[topic_id] = ""
            junk_topics.append(topic_id)
            continue

        # Build the label
        label = ", ".join(keywords)
        new_labels[topic_id] = label

    return new_labels, junk_topics