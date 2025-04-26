# src/utils/build_trend_data.py

import pandas as pd
import json
import os

def build_trend_data(
    topics_over_time_file="../../outputs/topics_over_time.csv",
    topic_labels_file="../../outputs/topic_labels.json",
    trend_data_output_file="../../data/trend_data.json"
):
    print("Building trend data...")

    # Pre-check
    if not os.path.exists(topics_over_time_file) or not os.path.exists(topic_labels_file):
        raise FileNotFoundError("Cannot build trend_data.json: missing topics_over_time.csv or topic_labels.json")

    # Load data
    topics_over_time_df = pd.read_csv(topics_over_time_file)
    with open(topic_labels_file, "r") as f:
        topic_labels = json.load(f)

    # Pivot
    pivot_df = topics_over_time_df.pivot_table(
        index='Topic', columns='Timestamp', values='Frequency', fill_value=0
    )

    # Build trend dictionary
    trend_data = {}
    for topic_num, row in pivot_df.iterrows():
        label = topic_labels.get(str(topic_num), f"Topic_{topic_num}")
        trend_data[label] = {str(year)[:4]: freq for year, freq in row.items()}
        
    # Save trend data
    os.makedirs(os.path.dirname(trend_data_output_file), exist_ok=True)
    with open(trend_data_output_file, "w") as f:
        json.dump(trend_data, f, indent=2)

    print(f"Trend data successfully saved to {trend_data_output_file}")

# Optional standalone run
if __name__ == "__main__":
    build_trend_data()
