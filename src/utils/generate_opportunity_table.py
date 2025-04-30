# ============================================================
# Function to match discovered topics to business opportunities 
# and recommend priorities based on size and growth.
#   - High size + High growth = High Priority
#   - High size + Low growth = Medium Priority
#   - Low size + High growth = Medium Priority
#   - Low size + Low growth = Low Priority
#
# Now supports using externally generated clean_labels mapping
# to populate the "Trend" column.
# Author: Katharine Leney, April 2025 (updated)
# ============================================================

import pandas as pd
from utils.opportunity_map import business_opportunities

def generate_opportunity_table(topic_model, topics_over_time_df, clean_labels=None, top_n_topics=None):

    # -------------------------------------------------
    # Extract topic info and counts
    # -------------------------------------------------
    topic_info = topic_model.get_topic_info()
    topic_df = topic_info.loc[topic_info["Topic"] != -1, ["Topic", "Name", "Count"]].copy()
    if top_n_topics is not None:
        topic_df = topic_df.nlargest(top_n_topics, "Count")
    total_count = topic_df["Count"].sum()

    # -------------------------------------------------
    # Compute topic growth
    # -------------------------------------------------
    #growth_dict = {}
    #for tid in topic_df["Topic"]:
    #    subset = topics_over_time_df[topics_over_time_df["Topic"] == tid]
    #    if subset.empty:
    #        growth = 0
    #    else:
    #        growth = subset.iloc[-1]["Frequency"] - subset.iloc[0]["Frequency"]
    #    growth_dict[tid] = growth

    # -------------------------------------------------
    # Compute topic growth based on % share per year
    # -------------------------------------------------
    # 0) detect which columns to use
    if "Year" in topics_over_time_df.columns:
        time_col = "Year"
    elif "year" in topics_over_time_df.columns:
        time_col = "year"
    else:
        raise KeyError(
            f"No 'Year' column found in topics_over_time_df; available: {topics_over_time_df.columns.tolist()}"
        )

    if "Frequency" in topics_over_time_df.columns:
        value_col = "Frequency"
    elif "Count" in topics_over_time_df.columns:
        value_col = "Count"
    else:
        raise KeyError(
            f"No 'Frequency' column found in topics_over_time_df; available: {topics_over_time_df.columns.tolist()}"
        )

    # 1) total keywords per year
    year_totals = topics_over_time_df\
        .groupby(time_col)[value_col]\
        .sum()\
        .to_dict()

    # 2) compute pct‐point growth for each topic
    growth_dict = {}
    for tid in topic_df["Topic"]:
        subset = (
            topics_over_time_df[topics_over_time_df["Topic"] == tid]
            .sort_values(time_col)
        )
        if subset.empty:
            growth = 0
        else:
            first_year, last_year = subset.iloc[0][time_col], subset.iloc[-1][time_col]
            first_val = subset.iloc[0][value_col]
            last_val  = subset.iloc[-1][value_col]
            # avoid zero‐division
            if year_totals.get(first_year, 0) == 0 or year_totals.get(last_year, 0) == 0:
                growth = 0
            else:
                growth = (
                    last_val  / year_totals[last_year]
                    - first_val / year_totals[first_year]
                ) * 100
        growth_dict[tid] = growth

    # -------------------------------------------------
    # Match to business opportunities
    # -------------------------------------------------
    matched = []
    unmatched = []
    for _, row in topic_df.iterrows():
        tid = row["Topic"]
        raw_label = row["Name"]
        # Use cleaned label if provided, else fall back to raw
        trend_label = clean_labels.get(tid, raw_label) if clean_labels else raw_label
        size = row["Count"]
        frac = size / total_count
        growth = growth_dict.get(tid, 0)

        found = False
        for keyword, info in business_opportunities.items():
            if keyword.lower() in raw_label.lower():
                # Priority logic
                if frac > 0.02 and growth > 5:
                    prio = "High"
                elif (frac > 0.02 and growth > 0) or growth > 1:
                    prio = "Medium"
                else:
                    prio = "Low"

                matched.append({
                    "Trend": trend_label,
                    "Detected Label": raw_label,
                    "Implication": info["Implication"],
                    "Ongoing Activity": info["Opportunity"],
                    "Priority (Recommended)": prio,
                    "Keyword Fraction (%)": round(frac * 100, 2),
                    "Topic Growth": round(growth, 4)
                })
                found = True
                break
        if not found:
            unmatched.append(raw_label)

    # -------------------------------------------------
    # Final DataFrame and sorting
    # -------------------------------------------------
    df = pd.DataFrame(matched)
    if df.empty:
        print("\n WARNING: No matching business opportunities found for detected topics.")
        print("\t ==> Check if topic labels and business_opportunities mapping is aligned.\n")
    else:
        order = {"High": 0, "Medium": 1, "Low": 2, "To be assessed": 3}
        df["_prio"] = df["Priority (Recommended)"].map(order)
        df = df.sort_values(by=["_prio", "Keyword Fraction (%)"], ascending=[True, False])
        df = df.drop(columns=["_prio"]).reset_index(drop=True)

    return df, unmatched
