import pandas as pd

# ==================================================================
# Function to filter Pandas dataframe by year
# ==================================================================

def filter_dataframe_by_year(df, min_year=None, max_year=None, exclude_years=None):

    if "Timestamp" in df.columns:
        df = df.copy()
        df["year"] = df["Timestamp"].dt.year
    elif "year" not in df.columns:
        raise ValueError("DataFrame must contain a 'Timestamp' or 'year' column.")

    mask = pd.Series([True] * len(df))
    if min_year is not None:
        mask &= df["year"] >= min_year
    if max_year is not None:
        mask &= df["year"] <= max_year
    if exclude_years:
        mask &= ~df["year"].isin(exclude_years)

    # Build human-readable description for plot labels
    desc_parts = []
    if min_year and max_year:
        desc_parts.append(f"Data from {min_year} to {max_year}")
    elif min_year:
        desc_parts.append(f"Data from {min_year} onwards")
    elif max_year:
        desc_parts.append(f"Data up to {max_year}")
    else:
        desc_parts.append("All years")

    if exclude_years:
        sorted_excludes = ", ".join(str(y) for y in sorted(exclude_years))
        desc_parts.append(f"(excluding {sorted_excludes})")

    description = " ".join(desc_parts)

    return df[mask], description

# ==================================================================
# Function to return the top_k most frequent 
# topics after filtering.
# ==================================================================
def get_top_topics(filtered_topics, top_k=10):
    
    # Group the filtered topics by year and topic, then count
    yearly_topic_counts = (
    filtered_topics.groupby(["Timestamp", "Topic"])
    .size()
    .unstack(fill_value=0)
    )

    # Normalise per year (row-wise)
    yearly_topic_freq = yearly_topic_counts.div(yearly_topic_counts.sum(axis=1), axis=0)

    # Average across all years
    avg_topic_freq = yearly_topic_freq.mean(axis=0)

    # Select top_k topics
    top_topics = avg_topic_freq.sort_values(ascending=False).head(top_k).index.tolist()
    return top_topics
